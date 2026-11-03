import {
  deleteFont,
  downloadFont,
  getFont,
  listFonts,
  uploadFont,
  validateFont,
  type FontRecord,
} from '@features/fonts';
import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';

import type { FontArchiveTransport, InvkFontDependency } from './fonts';

import { FontImportQuotaError } from './format';

/** Resolves identity collisions across servers by exact bytes, never by a family name. */
const resolveFont = async (dependency: InvkFontDependency, signal?: AbortSignal): Promise<FontRecord> => {
  for (const id of dependency.references) {
    try {
      const font = await getFont(id, signal);
      if (font.contentHash === dependency.contentHash) {
        return font;
      }
    } catch (error) {
      if (!(error instanceof ApiError) || error.status !== 404) {
        throw error;
      }
    }
  }
  const page = await listFonts({ contentHash: dependency.contentHash, limit: 1 }, signal);
  const font = page.items.find((item) => item.contentHash === dependency.contentHash);
  if (font) {
    return font;
  }
  throw new Error(`The font “${dependency.label}” is unavailable. Install it or export without font files.`);
};

export const createFontArchiveTransport = (): FontArchiveTransport => {
  const owner = captureAccountScope();
  return {
    download: async (dependency, signal = owner.signal) => {
      assertAccountScopeCurrent(owner);
      const font = await resolveFont(dependency, signal);
      const bytes = await downloadFont({ contentHash: dependency.contentHash, id: font.id }, signal);
      assertAccountScopeCurrent(owner);
      return { bytes, filename: font.filename };
    },
    remove: async (id, signal = owner.signal) => {
      assertAccountScopeCurrent(owner);
      await deleteFont(id, signal);
      assertAccountScopeCurrent(owner);
    },
    upload: async (file, signal = owner.signal) => {
      assertAccountScopeCurrent(owner);
      try {
        const result = await uploadFont(file, 'private', signal);
        assertAccountScopeCurrent(owner);
        return {
          created: result.created,
          font: {
            contentHash: result.font.contentHash,
            family: result.font.family,
            id: result.font.id,
            label: result.font.label,
          },
        };
      } catch (error) {
        if (error instanceof ApiError && error.status === 413) {
          throw new FontImportQuotaError();
        }
        throw error;
      }
    },
    validate: async (file, signal = owner.signal) => {
      assertAccountScopeCurrent(owner);
      try {
        await validateFont(file, signal);
        assertAccountScopeCurrent(owner);
      } catch (error) {
        if (error instanceof ApiError && error.status === 413) {
          throw new FontImportQuotaError();
        }
        throw error;
      }
    },
  };
};
