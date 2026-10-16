/**
 * OpenCode v2 Client Wrapper
 *
 * Implements the OpenCodeClient interface against the OpenCode 2.x server
 * (`/api/*` routes) using the official `@opencode/client` Promise client.
 *
 * Differences to v1 that callers can observe:
 * - Prompts are asynchronous on the server; `prompt` submits the message,
 *   waits until the session is idle and then collects the assistant reply.
 * - Nobody can answer permission requests or questions over MCP, so they are
 *   rejected with feedback the model can see (like `opencode run` does).
 * - Text search, symbol search, sharing and runtime config updates have no
 *   v2 endpoint and fail with NOT_SUPPORTED.
 */

import { readFile } from 'fs/promises';
import { basename, resolve } from 'path';
import { OpenCode } from '@opencode/client';
import type { ServerConfig } from '../utils/types.js';
import { OpenCodeMCPError, ErrorCodes } from '../utils/types.js';
import type { OpenCodeClient, PromptOptions } from './opencode.js';

type V2Client = ReturnType<typeof OpenCode.make>;
type Location = { directory: string } | undefined;

const PERMISSION_REJECTED_FEEDBACK =
  'This request came in through MCP, where nobody can approve permissions, so the request was rejected. Continue without this action.';
const QUESTION_CANCELLED_FEEDBACK =
  'This request came in through MCP, where nobody can answer questions. Continue with your best judgement.';

const POLL_INTERVAL_MS = 1000;
const ATTACH_FILE_MAX_BYTES = 10 * 1024 * 1024;

function notSupported(operation: string, hint: string): OpenCodeMCPError {
  return new OpenCodeMCPError(
    `${operation}: not supported by OpenCode 2.x servers. ${hint}`,
    ErrorCodes.NOT_SUPPORTED
  );
}

/**
 * Detect whether the server at `serverUrl` speaks the v2 API.
 * v2 answers `/api/info` with JSON (401 when a password is required);
 * v1 servers serve their web UI or 404 there.
 */
export async function isV2Server(serverUrl: string, headers?: Record<string, string>): Promise<boolean> {
  try {
    const res = await fetch(new URL('/api/info', serverUrl), {
      headers,
      signal: AbortSignal.timeout(5000),
    });
    if (!res.headers.get('content-type')?.includes('application/json')) return false;
    if (res.status === 401) return true;
    const info = await res.json() as { version?: unknown };
    return typeof info.version === 'string' && info.version.startsWith('2.');
  } catch {
    return false;
  }
}

export function authHeaders(config: ServerConfig): Record<string, string> | undefined {
  if (!config.serverPassword) return undefined;
  const credentials = `${config.serverUsername ?? 'opencode'}:${config.serverPassword}`;
  return { authorization: 'Basic ' + Buffer.from(credentials).toString('base64') };
}

export async function createV2Client(config: ServerConfig): Promise<OpenCodeClient> {
  const defaultDirectory = process.env.OPENCODE_DEFAULT_PROJECT;
  const client: V2Client = OpenCode.make({
    baseUrl: config.serverUrl,
    headers: authHeaders(config),
  });

  let connected = false;
  const timeout = config.timeout ?? 120000;
  const promptTimeout = config.promptTimeout ?? 1800000;
  const location = (directory?: string): Location => {
    const effective = directory ?? defaultDirectory;
    return effective ? { directory: effective } : undefined;
  };

  /**
   * Execute an async operation with timeout
   */
  async function withTimeout<T>(
    operation: (signal: AbortSignal) => Promise<T>,
    operationName: string,
    timeoutMs = timeout
  ): Promise<T> {
    const signal = AbortSignal.timeout(timeoutMs);
    try {
      return await operation(signal);
    } catch (error) {
      if (signal.aborted) {
        throw new OpenCodeMCPError(
          `${operationName} timed out after ${timeoutMs}ms`,
          ErrorCodes.TIMEOUT
        );
      }
      throw error;
    }
  }

  async function ensureConnected(): Promise<void> {
    if (!connected) {
      const healthy = await isHealthy();
      if (!healthy) {
        throw new OpenCodeMCPError(
          `Cannot connect to OpenCode server at ${config.serverUrl}`,
          ErrorCodes.CONNECTION_FAILED
        );
      }
      connected = true;
    }
  }

  async function isHealthy(): Promise<boolean> {
    try {
      await withTimeout(signal => client.server.info({ signal }), 'Health check');
      return true;
    } catch {
      return false;
    }
  }

  async function listSessions(): Promise<unknown[]> {
    await ensureConnected();
    const result = await withTimeout(
      signal => client.session.list({ limit: 100, order: 'desc' }, { signal }),
      'List sessions'
    );
    return result.data.map(session => ({ ...session, directory: session.location?.directory }));
  }

  async function createSession(
    title?: string,
    model?: { providerID: string; modelID: string },
    directory?: string
  ): Promise<{ id: string; title?: string; directory?: string }> {
    await ensureConnected();
    const session = await withTimeout(
      signal => client.session.create({
        title,
        location: location(directory),
        ...(model && { model: { providerID: model.providerID, id: model.modelID } }),
      }, { signal }),
      'Create session'
    );
    return {
      id: session.id,
      title: session.title,
      directory: session.location?.directory,
    };
  }

  async function getSession(id: string): Promise<unknown> {
    await ensureConnected();
    return withTimeout(signal => client.session.get({ sessionID: id }, { signal }), 'Get session');
  }

  async function getSessionMessage(sessionId: string, messageId: string): Promise<{ info?: unknown; parts?: unknown[] }> {
    await ensureConnected();
    const message = await withTimeout(
      signal => client.session.message.get({ sessionID: sessionId, messageID: messageId }, { signal }),
      'Get session message'
    );
    // Map onto the v1 message shape the tool handlers understand
    const error = 'error' in message ? message.error : undefined;
    return {
      info: {
        ...message,
        role: message.type,
        ...(error && { error: { name: error.type, data: { message: error.message } } }),
      },
      parts: message.type === 'assistant' ? [...message.content] : [],
    };
  }

  async function abortSession(id: string): Promise<boolean> {
    await ensureConnected();
    await withTimeout(signal => client.session.interrupt({ sessionID: id }, { signal }), 'Abort session');
    return true;
  }

  async function shareSession(_id: string): Promise<{ id: string; shareUrl?: string }> {
    throw notSupported('Session sharing', 'Share the session from the OpenCode app instead.');
  }

  /**
   * Reject everything in this session that waits for a human, so the run
   * can continue instead of blocking forever.
   */
  async function rejectPendingRequests(sessionId: string): Promise<void> {
    const [permissions, forms] = await Promise.all([
      client.permission.list({ sessionID: sessionId }).catch(() => []),
      client.session.form.list({ sessionID: sessionId }).catch(() => []),
    ]);
    await Promise.all([
      ...permissions.map(request => {
        console.error(`[prompt] Rejecting permission request: ${request.action} (${request.resources.join(', ')})`);
        return client.permission.reply({
          sessionID: request.sessionID,
          requestID: request.id,
          decision: 'reject',
          message: PERMISSION_REJECTED_FEEDBACK,
        }).catch(() => undefined);
      }),
      ...forms.map(form => client.session.form.cancel({
        sessionID: form.sessionID,
        formID: form.id,
        message: QUESTION_CANCELLED_FEEDBACK,
      }).catch(() => undefined)),
    ]);
  }

  /**
   * Messages newer than `messageId`, oldest first.
   */
  async function messagesAfter(sessionId: string, messageId: string) {
    const messages = [];
    let cursor: string | undefined;
    for (;;) {
      const page = await client.message.list(
        cursor ? { sessionID: sessionId, limit: 200, cursor } : { sessionID: sessionId, limit: 200, order: 'desc' }
      );
      for (const message of page.data) {
        if (message.id === messageId) return messages.reverse();
        messages.push(message);
      }
      cursor = page.cursor.next ?? undefined;
      if (!cursor) return messages.reverse();
    }
  }

  async function prepareFiles(files: string[], directory?: string) {
    const root = directory ?? defaultDirectory ?? process.cwd();
    const texts: string[] = [];
    const attachments: { uri: string; name: string }[] = [];
    for (const file of files) {
      const path = resolve(root, file);
      const content = await readFile(path).catch((error: unknown) => {
        throw new OpenCodeMCPError(`File not found: ${file}`, ErrorCodes.INVALID_INPUT, error as Error);
      });
      if (content.length > ATTACH_FILE_MAX_BYTES) {
        throw new OpenCodeMCPError(`Cannot attach a file larger than 10 MiB: ${file}`, ErrorCodes.INVALID_INPUT);
      }
      const text = content.toString('utf8');
      if (!content.includes(0) && Buffer.from(text, 'utf8').equals(content)) {
        texts.push(`<file name="${basename(path)}">\n${text}\n</file>`);
      } else {
        attachments.push({ uri: `data:application/octet-stream;base64,${content.toString('base64')}`, name: basename(path) });
      }
    }
    return { texts, attachments };
  }

  async function prompt(
    sessionId: string,
    promptText: string,
    options?: PromptOptions
  ): Promise<{ sessionId: string; messageId: string; content: string }> {
    await ensureConnected();

    if (options?.agent) {
      await withTimeout(
        signal => client.session.switchAgent({ sessionID: sessionId, agent: options.agent! }, { signal }),
        'Switch agent'
      );
    }
    if (options?.model) {
      const model = { providerID: options.model.providerID, id: options.model.modelID };
      await withTimeout(
        signal => client.session.switchModel({ sessionID: sessionId, model }, { signal }),
        'Switch model'
      );
    }

    const session = await client.session.get({ sessionID: sessionId });
    const { texts, attachments } = await prepareFiles(options?.files ?? [], session.location?.directory);
    const text = [promptText, ...texts].join('\n\n');

    if (options?.noReply) {
      // Context only: without resume the message waits in the inbox until the next prompt
      const message = await withTimeout(
        signal => client.session.synthetic({ sessionID: sessionId, text, resume: false }, { signal }),
        'Add context'
      );
      return { sessionId, messageId: message.id, content: '' };
    }

    const user = await withTimeout(
      signal => client.session.prompt({ sessionID: sessionId, text, files: attachments, delivery: 'steer' }, { signal }),
      'Send prompt'
    );

    // Wait until an idle marker follows our message, answering nothing that needs a human
    const deadline = Date.now() + promptTimeout;
    for (;;) {
      await rejectPendingRequests(sessionId);
      const messages = await messagesAfter(sessionId, user.id);
      const idle = messages.find(message => message.type === 'idle');
      if (idle) {
        const assistant = messages.filter(message => message.type === 'assistant');
        const last = assistant.at(-1);
        const content = last
          ? last.content.flatMap(part => (part.type === 'text' ? [part.text] : [])).join('')
          : '';
        if (idle.outcome === 'failed' && !content) {
          throw new OpenCodeMCPError(
            idle.error?.message ?? 'OpenCode reported a failed run',
            ErrorCodes.TOOL_EXECUTION_FAILED
          );
        }
        return { sessionId, messageId: last?.id ?? '', content };
      }
      if (Date.now() > deadline) {
        await client.session.interrupt({ sessionID: sessionId }).catch(() => undefined);
        throw new OpenCodeMCPError(
          `Prompt timed out after ${promptTimeout}ms; the session was interrupted`,
          ErrorCodes.TIMEOUT
        );
      }
      // session.wait returns once the session is idle; bound it so pending requests are still polled
      await client.session.wait({ sessionID: sessionId }, { signal: AbortSignal.timeout(POLL_INTERVAL_MS * 5) })
        .catch(() => new Promise(r => setTimeout(r, POLL_INTERVAL_MS)));
    }
  }

  async function listAgents(): Promise<unknown[]> {
    await ensureConnected();
    const result = await withTimeout(
      signal => client.agent.list({ location: location() }, { signal }),
      'List agents'
    );
    // v1 used the agent identifier as `name`
    return result.data.map(agent => ({ ...agent, name: agent.id, displayName: agent.name }));
  }

  async function listProviders(): Promise<{ providers: unknown[]; defaults: Record<string, string> }> {
    await ensureConnected();
    const [providers, models, fallback] = await withTimeout(
      signal => Promise.all([
        client.provider.list({ location: location() }, { signal }),
        client.model.list({ location: location() }, { signal }),
        client.model.default({ location: location() }, { signal }).catch(() => undefined),
      ]),
      'List providers'
    );
    return {
      providers: providers.data.map(provider => ({
        ...provider,
        models: models.data
          .filter(model => model.providerID === provider.id)
          .map(model => ({ id: model.id, name: model.name })),
      })),
      defaults: fallback?.data ? { [fallback.data.providerID]: fallback.data.id } : {},
    };
  }

  async function updateConfig(_config: Record<string, unknown>): Promise<unknown> {
    throw notSupported(
      'Runtime config updates',
      'Edit opencode.json (or the file in OPENCODE_CONFIG) and reload the OpenCode server.'
    );
  }

  async function getConfig(): Promise<unknown> {
    await ensureConnected();
    return withTimeout(signal => client.config.get({ location: location() }, { signal }), 'Get config');
  }

  async function readFileContent(path: string): Promise<{ content: string }> {
    await ensureConnected();
    const bytes = await withTimeout(
      signal => client.file.read({ location: location(), path }, { signal }),
      'Read file'
    );
    return { content: Buffer.from(bytes).toString('utf8') };
  }

  async function searchText(_pattern: string, _directory?: string): Promise<unknown[]> {
    throw notSupported('Text search', 'Use opencode_run or opencode_agent_delegate to search the codebase.');
  }

  async function findFiles(
    query: string,
    type?: 'file' | 'directory',
    limit?: number
  ): Promise<string[]> {
    await ensureConnected();
    const result = await withTimeout(
      signal => client.file.find({
        location: location(),
        query,
        ...(type && { type }),
        ...(limit && { limit }),
      }, { signal }),
      'Find files'
    );
    return result.data.map(entry => entry.path);
  }

  async function findSymbols(_query: string): Promise<unknown[]> {
    throw notSupported('Symbol search', 'Use opencode_run or opencode_agent_delegate to look up symbols.');
  }

  function disconnect(): void {
    connected = false;
  }

  return {
    isHealthy,
    listSessions,
    createSession,
    getSession,
    getSessionMessage,
    abortSession,
    shareSession,
    prompt,
    listAgents,
    listProviders,
    updateConfig,
    getConfig,
    readFile: readFileContent,
    searchText,
    findFiles,
    findSymbols,
    disconnect,
  };
}
