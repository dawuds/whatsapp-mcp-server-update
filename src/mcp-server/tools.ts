import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import type { WhatsAppClient, WhatsAppChat } from './whatsapp.js';
import {
  GetMessagesInputSchema,
  ExportChatInputSchema,
  SearchMessagesInputSchema,
  GroupInfoInputSchema,
  SendMessageInputSchema,
  ReplyToMessageInputSchema,
  ListChatsInputSchema,
  GetDMMessagesInputSchema,
  SendDMInputSchema,
  ReplyToDMInputSchema,
  SearchDMsInputSchema,
  AnalyzeDMInputSchema,
  AnalyzeGroupInputSchema,
  type WhatsAppGroup,
} from './types.js';
import { analyzeChat, crossGroupSynthesis } from '../processor/analyzer.js';
import { formatBriefing } from '../processor/briefing.js';
import type { UserContext } from '../processor/analyzer.js';

// ── Fuzzy Group Name Matching ───────────────────────────────────────────────

function normalize(input: string): string {
  return input
    .toLowerCase()
    .replace(/[^\w\s]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function levenshtein(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () =>
    Array(n + 1).fill(0),
  );

  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + cost,
      );
    }
  }

  return dp[m][n];
}

async function resolveGroup(
  client: WhatsAppClient,
  groupName: string,
): Promise<WhatsAppGroup | null> {
  const groups = await client.getGroups();
  const needle = normalize(groupName);

  // 1. Exact match (after normalization)
  const exact = groups.find((g) => normalize(g.name) === needle);
  if (exact) return exact;

  // 2. Substring includes
  const includes = groups.filter((g) => normalize(g.name).includes(needle));
  if (includes.length === 1) return includes[0];
  if (includes.length > 1) {
    includes.sort((a, b) => a.name.length - b.name.length);
    return includes[0];
  }

  // 3. Levenshtein distance < 3
  let bestMatch: WhatsAppGroup | null = null;
  let bestDistance = Infinity;

  for (const group of groups) {
    const distance = levenshtein(normalize(group.name), needle);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestMatch = group;
    }
  }

  if (bestDistance < 3 && bestMatch) return bestMatch;

  return null;
}

async function resolveGroupOrError(
  client: WhatsAppClient,
  groupName: string,
): Promise<
  | { group: WhatsAppGroup; error?: undefined }
  | { group?: undefined; error: { type: 'text'; text: string } }
> {
  const group = await resolveGroup(client, groupName);
  if (!group) {
    const groups = await client.getGroups();
    const available = groups.map((g) => g.name).join(', ');
    return {
      error: {
        type: 'text' as const,
        text: `No group found matching "${groupName}". Available groups: ${available}`,
      },
    };
  }
  return { group };
}

// ── DM Contact Resolution ───────────────────────────────────────────────────

async function resolveDMContact(
  client: WhatsAppClient,
  nameOrNumber: string,
): Promise<WhatsAppChat | null> {
  const chats = await client.getDirectChats();
  const norm = normalize(nameOrNumber);
  const numOnly = nameOrNumber.replace(/\D/g, '');

  // 1. Exact name match
  const exact = chats.find((c) => normalize(c.name) === norm);
  if (exact) return exact;

  // 2. Phone number suffix match (7+ digits)
  if (numOnly.length >= 7) {
    const byNumber = chats.find((c) => c.jid.replace('@s.whatsapp.net', '').endsWith(numOnly));
    if (byNumber) return byNumber;
  }

  // 3. Substring
  const sub = chats.find(
    (c) => normalize(c.name).includes(norm) || norm.includes(normalize(c.name)),
  );
  if (sub) return sub;

  // 4. Levenshtein < 3
  let best: WhatsAppChat | null = null;
  let bestDist = Infinity;
  for (const c of chats) {
    const d = levenshtein(normalize(c.name), norm);
    if (d < bestDist) { bestDist = d; best = c; }
  }
  return bestDist < 3 ? best : null;
}

async function resolveDMOrError(
  client: WhatsAppClient,
  nameOrNumber: string,
): Promise<
  | { chat: WhatsAppChat; error?: undefined }
  | { chat?: undefined; error: { type: 'text'; text: string } }
> {
  const chat = await resolveDMContact(client, nameOrNumber);
  if (!chat) {
    const chats = await client.getDirectChats();
    const names = chats.slice(0, 20).map((c) => c.name).join(', ');
    return {
      error: {
        type: 'text' as const,
        text: `No DM contact found matching "${nameOrNumber}". Recent contacts: ${names}`,
      },
    };
  }
  return { chat };
}

// ── Analysis Helpers ────────────────────────────────────────────────────────

const DEFAULT_CONTEXT: UserContext = {
  name: 'User',
  aliases: [],
  role: 'WhatsApp user',
  focusAreas: ['technology', 'business'],
  opportunityTypes: ['collaboration', 'learning'],
  contentOutlets: [],
};

function loadUserContext(): UserContext {
  const configPath = join(process.cwd(), 'config', 'user-context.json');
  try {
    if (existsSync(configPath)) {
      const raw = readFileSync(configPath, 'utf-8');
      return JSON.parse(raw) as UserContext;
    }
  } catch {
    // fall through to default
  }
  return DEFAULT_CONTEXT;
}

function messagesToParsed(messages: Awaited<ReturnType<WhatsAppClient['getGroupMessages']>>) {
  return messages.map((m) => ({
    platform: 'whatsapp' as const,
    timestamp: new Date(m.timestamp * 1000),
    author: m.authorName || m.author,
    body: m.body,
    isMedia: m.hasMedia,
    isSystem: false,
  }));
}

// ── Tool Registration ───────────────────────────────────────────────────────

export function registerTools(server: Server, client: WhatsAppClient): void {
  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: [
      {
        name: 'whatsapp_list_groups',
        description:
          'List all WhatsApp groups the authenticated user belongs to, sorted by most recent activity.',
        inputSchema: {
          type: 'object' as const,
          properties: {},
          required: [],
        },
      },
      {
        name: 'whatsapp_get_messages',
        description:
          'Get messages from a WhatsApp group. Supports fuzzy group name matching, message count limit, and date range filtering.',
        inputSchema: {
          type: 'object' as const,
          properties: {
            groupName: {
              type: 'string',
              description: 'Name of the group (fuzzy-matched)',
            },
            limit: {
              type: 'number',
              description: 'Maximum messages to return (default: 200, max: 500)',
              default: 200,
            },
            afterDate: {
              type: 'string',
              description: 'Only messages after this date (YYYY-MM-DD)',
            },
            beforeDate: {
              type: 'string',
              description: 'Only messages before this date (YYYY-MM-DD)',
            },
          },
          required: ['groupName'],
        },
      },
      {
        name: 'whatsapp_export_chat',
        description:
          'Export a WhatsApp group chat as .txt in the same format as WhatsApp built-in export.',
        inputSchema: {
          type: 'object' as const,
          properties: {
            groupName: {
              type: 'string',
              description: 'Name of the group (fuzzy-matched)',
            },
            limit: {
              type: 'number',
              description: 'Maximum messages to include (default: 500, max: 500)',
              default: 500,
            },
          },
          required: ['groupName'],
        },
      },
      {
        name: 'whatsapp_search_messages',
        description:
          'Search messages containing a keyword or phrase, optionally scoped to a specific group.',
        inputSchema: {
          type: 'object' as const,
          properties: {
            query: {
              type: 'string',
              description: 'Search query (keyword or phrase)',
            },
            groupName: {
              type: 'string',
              description: 'Optional: limit search to a specific group (fuzzy-matched)',
            },
            limit: {
              type: 'number',
              description: 'Maximum results to return (default: 50, max: 200)',
              default: 50,
            },
          },
          required: ['query'],
        },
      },
      {
        name: 'whatsapp_group_info',
        description:
          'Get detailed metadata about a WhatsApp group including description, participants, and creation date.',
        inputSchema: {
          type: 'object' as const,
          properties: {
            groupName: {
              type: 'string',
              description: 'Name of the group (fuzzy-matched)',
            },
          },
          required: ['groupName'],
        },
      },
      {
        name: 'whatsapp_send_message',
        description:
          'Send a message to a WhatsApp group.',
        inputSchema: {
          type: 'object' as const,
          properties: {
            groupName: {
              type: 'string',
              description: 'Name of the group (fuzzy-matched)',
            },
            message: {
              type: 'string',
              description: 'Message text to send',
            },
          },
          required: ['groupName', 'message'],
        },
      },
      {
        name: 'whatsapp_reply_to_message',
        description:
          'Reply to a specific message in a WhatsApp group. The reply will be shown as a quoted reply.',
        inputSchema: {
          type: 'object' as const,
          properties: {
            groupName: { type: 'string', description: 'Name of the group (fuzzy-matched)' },
            messageId: { type: 'string', description: 'The ID of the message to reply to' },
            message: { type: 'string', description: 'Reply text to send' },
          },
          required: ['groupName', 'messageId', 'message'],
        },
      },
      {
        name: 'whatsapp_analyze_group',
        description:
          'Run intelligence analysis on a WhatsApp group: extracts themes, big ideas, opportunities, participation, live threads, and notable quotes.',
        inputSchema: {
          type: 'object' as const,
          properties: {
            groupName: { type: 'string', description: 'Name of the group (fuzzy-matched)' },
            limit: { type: 'number', description: 'Messages to analyse (default: 300, max: 500)', default: 300 },
            afterDate: { type: 'string', description: 'Only messages after this date (YYYY-MM-DD)' },
            beforeDate: { type: 'string', description: 'Only messages before this date (YYYY-MM-DD)' },
          },
          required: ['groupName'],
        },
      },
      // ── DM Tools ────────────────────────────────────────────────────────
      {
        name: 'whatsapp_list_chats',
        description: 'List direct message (DM) conversations sorted by most recent activity.',
        inputSchema: {
          type: 'object' as const,
          properties: {
            limit: { type: 'number', description: 'Max contacts to return (default: 50)', default: 50 },
          },
          required: [],
        },
      },
      {
        name: 'whatsapp_get_dm_messages',
        description: 'Get messages from a direct message conversation. Supports fuzzy contact name matching and date range filtering.',
        inputSchema: {
          type: 'object' as const,
          properties: {
            contactName: { type: 'string', description: 'Contact name or phone number (fuzzy-matched)' },
            limit: { type: 'number', description: 'Max messages to return (default: 100)', default: 100 },
            afterDate: { type: 'string', description: 'Only messages after this date (YYYY-MM-DD)' },
            beforeDate: { type: 'string', description: 'Only messages before this date (YYYY-MM-DD)' },
          },
          required: ['contactName'],
        },
      },
      {
        name: 'whatsapp_send_dm',
        description: 'Send a direct message to a WhatsApp contact. Uses fuzzy name matching.',
        inputSchema: {
          type: 'object' as const,
          properties: {
            contactName: { type: 'string', description: 'Contact name or phone number (fuzzy-matched)' },
            message: { type: 'string', description: 'Message text to send' },
          },
          required: ['contactName', 'message'],
        },
      },
      {
        name: 'whatsapp_reply_to_dm',
        description: 'Send a quoted reply to a specific DM message.',
        inputSchema: {
          type: 'object' as const,
          properties: {
            contactName: { type: 'string', description: 'Contact name or phone number (fuzzy-matched)' },
            messageId: { type: 'string', description: 'The ID of the message to reply to' },
            message: { type: 'string', description: 'Reply text to send' },
          },
          required: ['contactName', 'messageId', 'message'],
        },
      },
      {
        name: 'whatsapp_search_dms',
        description: 'Search messages across all DM conversations, or within a specific contact thread.',
        inputSchema: {
          type: 'object' as const,
          properties: {
            query: { type: 'string', description: 'Search keyword or phrase' },
            contactName: { type: 'string', description: 'Optional: limit search to one contact' },
            limit: { type: 'number', description: 'Max results (default: 50)', default: 50 },
          },
          required: ['query'],
        },
      },
      {
        name: 'whatsapp_analyze_dm',
        description: 'Run intelligence analysis on a DM conversation thread.',
        inputSchema: {
          type: 'object' as const,
          properties: {
            contactName: { type: 'string', description: 'Contact name or phone number (fuzzy-matched)' },
            limit: { type: 'number', description: 'Messages to analyse (default: 100)', default: 100 },
            afterDate: { type: 'string', description: 'Only messages after this date (YYYY-MM-DD)' },
            beforeDate: { type: 'string', description: 'Only messages before this date (YYYY-MM-DD)' },
          },
          required: ['contactName'],
        },
      },
    ],
  }));

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;

    // Retry helper — one automatic retry on transient errors (Puppeteer crashes,
    // WhatsApp Web flakiness). Waits 3s before retry to let the browser stabilize.
    const MAX_ATTEMPTS = 2;
    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    try {
      switch (name) {
        case 'whatsapp_list_groups': {
          const groups = await client.getGroups();
          const sorted = [...groups].sort(
            (a, b) => b.lastActivityTimestamp - a.lastActivityTimestamp,
          );

          const formatted = sorted
            .map((g, i) => {
              const date = new Date(g.lastActivityTimestamp * 1000).toISOString();
              const preview = g.lastMessage
                ? g.lastMessage.slice(0, 80)
                : '(no messages)';
              return `${i + 1}. ${g.name} (${g.memberCount} members)\n   Last active: ${date}\n   Last message: ${preview}`;
            })
            .join('\n\n');

          return {
            content: [{ type: 'text' as const, text: `Found ${sorted.length} groups:\n\n${formatted}` }],
          };
        }

        case 'whatsapp_get_messages': {
          const parsed = GetMessagesInputSchema.parse(args);
          const result = await resolveGroupOrError(client, parsed.groupName);
          if (result.error) return { content: [result.error], isError: true };

          const options: { limit: number; after?: number; before?: number } = {
            limit: parsed.limit,
          };

          if (parsed.afterDate) {
            options.after = Math.floor(new Date(parsed.afterDate + 'T00:00:00Z').getTime() / 1000);
          }
          if (parsed.beforeDate) {
            options.before = Math.floor(new Date(parsed.beforeDate + 'T23:59:59Z').getTime() / 1000);
          }

          const messages = await client.getGroupMessages(result.group.id, options);

          const formatted = messages
            .map((m) => {
              const date = new Date(m.timestamp * 1000).toISOString();
              const media = m.hasMedia ? ' [media]' : '';
              const fwd = m.isForwarded ? ' [forwarded]' : '';
              const quote = m.quotedMsg
                ? `\n   > ${m.quotedMsg.author}: ${m.quotedMsg.body.slice(0, 60)}`
                : '';
              return `[${date}] [id:${m.id}] ${m.authorName}${media}${fwd}: ${m.body}${quote}`;
            })
            .join('\n');

          return {
            content: [{
              type: 'text' as const,
              text: `${messages.length} messages from "${result.group.name}":\n\n${formatted}`,
            }],
          };
        }

        case 'whatsapp_export_chat': {
          const parsed = ExportChatInputSchema.parse(args);
          const result = await resolveGroupOrError(client, parsed.groupName);
          if (result.error) return { content: [result.error], isError: true };

          const exported = await client.exportChat(result.group.id, parsed.limit);
          return { content: [{ type: 'text' as const, text: exported }] };
        }

        case 'whatsapp_search_messages': {
          const parsed = SearchMessagesInputSchema.parse(args);

          let groupId: string | undefined;
          if (parsed.groupName) {
            const result = await resolveGroupOrError(client, parsed.groupName);
            if (result.error) return { content: [result.error], isError: true };
            groupId = result.group.id;
          }

          const messages = await client.searchMessages(parsed.query, groupId);
          const limited = messages.slice(0, parsed.limit);

          const formatted = limited
            .map((m) => {
              const date = new Date(m.timestamp * 1000).toISOString();
              return `[${date}] [id:${m.id}] ${m.authorName}: ${m.body}`;
            })
            .join('\n');

          const scope = parsed.groupName ? `in "${parsed.groupName}"` : 'across all groups';

          return {
            content: [{
              type: 'text' as const,
              text: `${limited.length} results for "${parsed.query}" ${scope}:\n\n${formatted}`,
            }],
          };
        }

        case 'whatsapp_group_info': {
          const parsed = GroupInfoInputSchema.parse(args);
          const result = await resolveGroupOrError(client, parsed.groupName);
          if (result.error) return { content: [result.error], isError: true };

          const info = await client.getGroupInfo(result.group.id);
          const admins = info.participants.filter((p) => p.isAdmin);
          const members = info.participants.filter((p) => !p.isAdmin);
          const created = new Date(info.createdAt * 1000).toISOString();

          const text = [
            `Group: ${info.name}`,
            `Description: ${info.description || '(none)'}`,
            `Created: ${created}`,
            `Total participants: ${info.participants.length}`,
            '',
            `Admins (${admins.length}):`,
            ...admins.map((a) => `  - ${a.name}`),
            '',
            `Members (${members.length}):`,
            ...members.map((m) => `  - ${m.name}`),
          ].join('\n');

          return { content: [{ type: 'text' as const, text }] };
        }

        case 'whatsapp_send_message': {
          const parsed = SendMessageInputSchema.parse(args);
          const result = await resolveGroupOrError(client, parsed.groupName);
          if (result.error) return { content: [result.error], isError: true };

          const sent = await client.sendMessage(result.group.id, parsed.message);
          return {
            content: [{
              type: 'text' as const,
              text: `Message sent to "${result.group.name}".\nMessage ID: ${sent.id}\nTimestamp: ${new Date(sent.timestamp * 1000).toISOString()}`,
            }],
          };
        }

        case 'whatsapp_reply_to_message': {
          const parsed = ReplyToMessageInputSchema.parse(args);
          const result = await resolveGroupOrError(client, parsed.groupName);
          if (result.error) return { content: [result.error], isError: true };

          const sent = await client.sendMessage(
            result.group.id,
            parsed.message,
            parsed.messageId,
          );
          return {
            content: [{
              type: 'text' as const,
              text: `Reply sent to "${result.group.name}" (quoting ${parsed.messageId}).\nMessage ID: ${sent.id}\nTimestamp: ${new Date(sent.timestamp * 1000).toISOString()}`,
            }],
          };
        }

        case 'whatsapp_analyze_group': {
          const parsed = AnalyzeGroupInputSchema.parse(args);
          const result = await resolveGroupOrError(client, parsed.groupName);
          if (result.error) return { content: [result.error], isError: true };

          const options: { limit: number; after?: number; before?: number } = { limit: parsed.limit };
          if (parsed.afterDate) options.after = Math.floor(new Date(parsed.afterDate + 'T00:00:00Z').getTime() / 1000);
          if (parsed.beforeDate) options.before = Math.floor(new Date(parsed.beforeDate + 'T23:59:59Z').getTime() / 1000);

          const messages = await client.getGroupMessages(result.group.id, options);
          if (messages.length === 0) {
            return { content: [{ type: 'text' as const, text: `No messages found in "${result.group.name}" for the given range.` }] };
          }

          const ctx = loadUserContext();
          const parsed2 = messagesToParsed(messages);
          const analysis = analyzeChat(parsed2, ctx, result.group.name);
          const briefing = formatBriefing(analysis, { sections: parsed.sections });
          return { content: [{ type: 'text' as const, text: briefing }] };
        }

        case 'whatsapp_list_chats': {
          const parsed = ListChatsInputSchema.parse(args);
          const chats = await client.getDirectChats();
          const limited = chats.slice(0, parsed.limit);
          const formatted = limited
            .map((c, i) => {
              const date = new Date(c.lastActivityTimestamp * 1000).toISOString();
              const preview = c.lastMessage ? c.lastMessage.slice(0, 60) : '(no messages)';
              return `${i + 1}. ${c.name}\n   Last active: ${date}\n   Last message: ${preview}`;
            })
            .join('\n\n');
          return {
            content: [{ type: 'text' as const, text: `Found ${chats.length} DM contacts (showing ${limited.length}):\n\n${formatted}` }],
          };
        }

        case 'whatsapp_get_dm_messages': {
          const parsed = GetDMMessagesInputSchema.parse(args);
          const result = await resolveDMOrError(client, parsed.contactName);
          if (result.error) return { content: [result.error], isError: true };

          const options: { limit: number; after?: number; before?: number } = { limit: parsed.limit };
          if (parsed.afterDate) options.after = Math.floor(new Date(parsed.afterDate + 'T00:00:00Z').getTime() / 1000);
          if (parsed.beforeDate) options.before = Math.floor(new Date(parsed.beforeDate + 'T23:59:59Z').getTime() / 1000);

          const messages = await client.getDMMessages(result.chat.jid, options);
          const formatted = messages
            .map((m) => {
              const date = new Date(m.timestamp * 1000).toISOString();
              const media = m.hasMedia ? ' [media]' : '';
              const quote = m.quotedMsg ? `\n   > ${m.quotedMsg.author}: ${m.quotedMsg.body.slice(0, 60)}` : '';
              return `[${date}] [id:${m.id}] ${m.authorName}${media}: ${m.body}${quote}`;
            })
            .join('\n');
          return {
            content: [{ type: 'text' as const, text: `${messages.length} messages with "${result.chat.name}":\n\n${formatted}` }],
          };
        }

        case 'whatsapp_send_dm': {
          const parsed = SendDMInputSchema.parse(args);
          const result = await resolveDMOrError(client, parsed.contactName);
          if (result.error) return { content: [result.error], isError: true };

          const sent = await client.sendMessage(result.chat.jid, parsed.message);
          return {
            content: [{ type: 'text' as const, text: `Message sent to "${result.chat.name}".\nMessage ID: ${sent.id}\nTimestamp: ${new Date(sent.timestamp * 1000).toISOString()}` }],
          };
        }

        case 'whatsapp_reply_to_dm': {
          const parsed = ReplyToDMInputSchema.parse(args);
          const result = await resolveDMOrError(client, parsed.contactName);
          if (result.error) return { content: [result.error], isError: true };

          const sent = await client.sendMessage(result.chat.jid, parsed.message, parsed.messageId);
          return {
            content: [{ type: 'text' as const, text: `Reply sent to "${result.chat.name}" (quoting ${parsed.messageId}).\nMessage ID: ${sent.id}\nTimestamp: ${new Date(sent.timestamp * 1000).toISOString()}` }],
          };
        }

        case 'whatsapp_search_dms': {
          const parsed = SearchDMsInputSchema.parse(args);
          let jid: string | undefined;
          if (parsed.contactName) {
            const result = await resolveDMOrError(client, parsed.contactName);
            if (result.error) return { content: [result.error], isError: true };
            jid = result.chat.jid;
          }

          const results = await client.searchDMs(parsed.query, jid, parsed.limit);
          const formatted = results
            .map((m) => {
              const date = new Date(m.timestamp * 1000).toISOString();
              return `[${date}] [${m.chatName}] [id:${m.id}] ${m.authorName}: ${m.body}`;
            })
            .join('\n');
          const scope = parsed.contactName ? `in "${parsed.contactName}"` : 'across all DMs';
          return {
            content: [{ type: 'text' as const, text: `${results.length} results for "${parsed.query}" ${scope}:\n\n${formatted}` }],
          };
        }

        case 'whatsapp_analyze_dm': {
          const parsed = AnalyzeDMInputSchema.parse(args);
          const result = await resolveDMOrError(client, parsed.contactName);
          if (result.error) return { content: [result.error], isError: true };

          const options: { limit: number; after?: number; before?: number } = { limit: parsed.limit };
          if (parsed.afterDate) options.after = Math.floor(new Date(parsed.afterDate + 'T00:00:00Z').getTime() / 1000);
          if (parsed.beforeDate) options.before = Math.floor(new Date(parsed.beforeDate + 'T23:59:59Z').getTime() / 1000);

          const messages = await client.getDMMessages(result.chat.jid, options);
          if (messages.length === 0) {
            return { content: [{ type: 'text' as const, text: `No messages found with "${result.chat.name}" for the given range.` }] };
          }

          const ctx = loadUserContext();
          const parsed2 = messagesToParsed(messages);
          const analysis = analyzeChat(parsed2, ctx, result.chat.name);
          const briefing = formatBriefing(analysis, { sections: parsed.sections });
          return { content: [{ type: 'text' as const, text: briefing }] };
        }

        default:
          return {
            content: [{ type: 'text' as const, text: `Unknown tool: ${name}` }],
            isError: true,
          };
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (attempt < MAX_ATTEMPTS) {
        // Log and retry after a short delay
        process.stderr.write(
          `[${new Date().toISOString()}] [tools] ${name} attempt ${attempt} failed: ${message} — retrying in 3s...\n`,
        );
        await new Promise((r) => setTimeout(r, 500));
        continue;
      }
      return {
        content: [{ type: 'text' as const, text: `Error executing ${name}: ${message}` }],
        isError: true,
      };
    }
    } // end retry loop

    // Unreachable — every path inside the loop returns or continues — but
    // TypeScript can't prove it, so we satisfy the return type here.
    return {
      content: [{ type: 'text' as const, text: `Error executing ${name}: max retries exceeded` }],
      isError: true as const,
    };
  });
}
