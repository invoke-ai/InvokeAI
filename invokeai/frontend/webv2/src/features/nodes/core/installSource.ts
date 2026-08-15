/**
 * Client-side mirror of the backend's install-source rules
 * (`invokeai/app/api/routers/custom_nodes.py`: `PACK_NAME_RE`,
 * `_extract_pack_name_from_source`, and the already-exists rejection), so a
 * doomed install is caught before the POST instead of surfacing as a failed
 * request.
 */

const PACK_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

export type InstallSourceIssue = 'empty' | 'invalidPackName' | 'alreadyInstalled';

export interface InstallSourceValidation {
  packName: string | null;
  /** null = installable. */
  issue: InstallSourceIssue | null;
}

/** The backend derives the pack directory name from the source's last path segment, minus `.git`. */
export const derivePackNameFromSource = (source: string): string | null => {
  let packName = source.trim().replace(/\/+$/, '').split('/').at(-1) ?? '';

  if (packName.endsWith('.git')) {
    packName = packName.slice(0, -'.git'.length);
  }

  // The regex also rejects '.', '..', backslashes, and empty names — the
  // backend's explicit pre-checks are subsumed by the fullmatch.
  return PACK_NAME_RE.test(packName) ? packName : null;
};

export const validateInstallSource = (
  source: string,
  installedPackNames: ReadonlySet<string>
): InstallSourceValidation => {
  const trimmed = source.trim();

  if (trimmed === '') {
    return { issue: 'empty', packName: null };
  }

  const packName = derivePackNameFromSource(trimmed);

  if (packName === null) {
    return { issue: 'invalidPackName', packName: null };
  }

  if (installedPackNames.has(packName)) {
    return { issue: 'alreadyInstalled', packName };
  }

  return { issue: null, packName };
};
