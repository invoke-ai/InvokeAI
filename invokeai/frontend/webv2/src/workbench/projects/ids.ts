import { createUuid } from '@platform/browser/randomUuid';
import { sha256 } from '@platform/browser/sha256';

export const createProjectId = (): string => `project-${createUuid()}`;

export const createDeterministicProjectId = async (scope: string): Promise<string> => {
  const digest = await sha256(new TextEncoder().encode(scope));
  const bytes = digest.slice(0, 16);
  bytes[6] = (bytes[6]! & 0x0f) | 0x80;
  bytes[8] = (bytes[8]! & 0x3f) | 0x80;
  const hex = [...bytes].map((byte) => byte.toString(16).padStart(2, '0')).join('');
  return `project-${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
};
