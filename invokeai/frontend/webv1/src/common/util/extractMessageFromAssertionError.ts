import type { AssertionError } from 'tsafe';

export function extractMessageFromAssertionError(error: AssertionError): string | null {
  const match = error.message.match(/Wrong assertion encountered: "(.*)"/);
  return match ? (match[1] ?? null) : null;
}
