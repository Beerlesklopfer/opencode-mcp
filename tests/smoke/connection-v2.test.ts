/**
 * Smoke Tests - Connection (OpenCode v2)
 *
 * Verifies the v2 client wrapper against a mocked `@opencode/client`.
 */

import { describe, test, expect, vi, beforeEach } from 'vitest';

type Message = { id: string; type: string; [key: string]: unknown };

const state: { messages: Message[]; permissions: unknown[] } = { messages: [], permissions: [] };

const mock = {
  server: { info: vi.fn().mockResolvedValue({ version: '2.0.30' }) },
  session: {
    list: vi.fn().mockResolvedValue({ data: [{ id: 'ses_1', location: { directory: '/repo' } }], cursor: {} }),
    create: vi.fn().mockResolvedValue({ id: 'ses_1', title: 'Test', location: { directory: '/repo' } }),
    get: vi.fn().mockResolvedValue({ id: 'ses_1', location: { directory: '/repo' } }),
    switchAgent: vi.fn().mockResolvedValue(undefined),
    switchModel: vi.fn().mockResolvedValue(undefined),
    prompt: vi.fn().mockResolvedValue({ id: 'msg_user', type: 'user' }),
    synthetic: vi.fn().mockResolvedValue({ id: 'msg_syn', type: 'synthetic' }),
    wait: vi.fn().mockResolvedValue(undefined),
    interrupt: vi.fn().mockResolvedValue(undefined),
    message: { get: vi.fn() },
    form: { list: vi.fn().mockResolvedValue([]), cancel: vi.fn() },
  },
  // Newest first, like the server with order=desc
  message: {
    list: vi.fn(async () => ({ data: [...state.messages].reverse(), cursor: { next: null } })),
  },
  permission: {
    list: vi.fn(async () => state.permissions.splice(0)),
    reply: vi.fn().mockResolvedValue(undefined),
  },
  agent: { list: vi.fn().mockResolvedValue({ data: [{ id: 'build', name: 'Build', mode: 'primary' }] }) },
  provider: { list: vi.fn().mockResolvedValue({ data: [{ id: 'lmstudio', name: 'LM Studio' }] }) },
  model: {
    list: vi.fn().mockResolvedValue({ data: [{ id: 'qwen', name: 'Qwen', providerID: 'lmstudio' }] }),
    default: vi.fn().mockResolvedValue({ data: { id: 'qwen', providerID: 'lmstudio' } }),
  },
  config: { get: vi.fn().mockResolvedValue([]) },
  file: {
    read: vi.fn().mockResolvedValue(new TextEncoder().encode('file content')),
    find: vi.fn().mockResolvedValue({ data: [{ path: 'src/a.ts', type: 'file' }] }),
  },
};

vi.mock('@opencode/client', () => ({
  OpenCode: { make: vi.fn(() => mock) },
}));

const config = {
  serverUrl: 'http://localhost:4096',
  autoStart: false,
  timeout: 30000,
  transport: 'stdio' as const,
  apiVersion: 'v2' as const,
};

async function client() {
  const { createClient } = await import('../../src/client/opencode.js');
  return createClient(config);
}

describe('Smoke Tests - Connection (v2)', () => {
  beforeEach(() => {
    state.messages = [{ id: 'msg_user', type: 'user' }];
    state.permissions = [];
    vi.clearAllMocks();
  });

  test('prompt waits for idle and returns the last assistant text', async () => {
    mock.session.wait.mockImplementationOnce(async () => {
      state.messages.push(
        { id: 'msg_a1', type: 'assistant', content: [{ type: 'text', text: 'thinking…' }] },
        { id: 'msg_a2', type: 'assistant', content: [{ type: 'text', text: 'Hello!' }] },
        { id: 'msg_idle', type: 'idle', outcome: 'succeeded' },
      );
    });
    const c = await client();
    const result = await c.prompt('ses_1', 'Hi', {
      agent: 'plan',
      model: { providerID: 'lmstudio', modelID: 'qwen' },
    });

    expect(result).toEqual({ sessionId: 'ses_1', messageId: 'msg_a2', content: 'Hello!' });
    expect(mock.session.switchAgent).toHaveBeenCalledWith({ sessionID: 'ses_1', agent: 'plan' }, expect.anything());
    expect(mock.session.switchModel).toHaveBeenCalledWith(
      { sessionID: 'ses_1', model: { providerID: 'lmstudio', id: 'qwen' } },
      expect.anything()
    );
  });

  test('pending permission requests are rejected with feedback', async () => {
    state.permissions = [{ id: 'per_1', sessionID: 'ses_1', action: 'read', resources: ['.env'] }];
    mock.session.wait.mockImplementationOnce(async () => {
      state.messages.push({ id: 'msg_idle', type: 'idle', outcome: 'succeeded' });
    });
    const c = await client();
    await c.prompt('ses_1', 'Read .env');

    expect(mock.permission.reply).toHaveBeenCalledWith(expect.objectContaining({
      requestID: 'per_1',
      decision: 'reject',
      message: expect.any(String),
    }));
  });

  test('a failed run without output raises the server error', async () => {
    state.messages.push({
      id: 'msg_idle',
      type: 'idle',
      outcome: 'failed',
      error: { type: 'provider.no-route', message: 'Model unavailable' },
    });
    const c = await client();
    await expect(c.prompt('ses_1', 'Hi')).rejects.toThrow('Model unavailable');
  });

  test('noReply adds context without starting a turn', async () => {
    const c = await client();
    const result = await c.prompt('ses_1', 'Context', { noReply: true });

    expect(result).toEqual({ sessionId: 'ses_1', messageId: 'msg_syn', content: '' });
    expect(mock.session.synthetic).toHaveBeenCalledWith(
      { sessionID: 'ses_1', text: 'Context', resume: false },
      expect.anything()
    );
    expect(mock.session.prompt).not.toHaveBeenCalled();
  });

  test('responses are mapped to the v1 shapes', async () => {
    const c = await client();

    expect(await c.listAgents()).toEqual([expect.objectContaining({ name: 'build', displayName: 'Build' })]);
    expect(await c.listProviders()).toEqual({
      providers: [expect.objectContaining({ id: 'lmstudio', models: [{ id: 'qwen', name: 'Qwen' }] })],
      defaults: { lmstudio: 'qwen' },
    });
    expect(await c.readFile('a.txt')).toEqual({ content: 'file content' });
    expect(await c.findFiles('a')).toEqual(['src/a.ts']);
    expect(await c.createSession('Test')).toEqual({ id: 'ses_1', title: 'Test', directory: '/repo' });
  });

  test('operations without a v2 endpoint report NOT_SUPPORTED', async () => {
    const c = await client();
    for (const call of [() => c.searchText('x'), () => c.findSymbols('x'), () => c.updateConfig({}), () => c.shareSession('ses_1')]) {
      await expect(call()).rejects.toMatchObject({ code: 'NOT_SUPPORTED' });
    }
  });
});
