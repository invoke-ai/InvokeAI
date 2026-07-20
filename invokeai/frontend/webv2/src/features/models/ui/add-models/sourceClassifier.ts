/** owner/repo with optional :variant[:path] qualifiers - the HF repo shape. */
const HF_REPO_PATTERN = /^[\w.-]+\/[\w.-]+(:[\w./-]*)*$/;
const FILE_EXTENSION_PATTERN = /\.[A-Za-z0-9]{1,10}$/;

export interface SourceKind {
  looksLocal: boolean;
  looksUrl: boolean;
  looksRepo: boolean;
  isInstallable: boolean;
  localKind: 'file' | 'folder' | null;
  /** Human label of the detected source kind, or null when it reads as a search. */
  label: string | null;
}

const classifyLocalPath = (value: string): 'file' | 'folder' => {
  if (/[\\/]$/.test(value)) {
    return 'folder';
  }

  const lastSegment = value.split(/[\\/]/).at(-1) ?? '';

  return FILE_EXTENSION_PATTERN.test(lastSegment) ? 'file' : 'folder';
};

/** Classify the box's text: a search term, or an installable source to Pull. */
export const classifySource = (value: string): SourceKind => {
  const looksLocal = value.startsWith('/') || /^[A-Za-z]:[\\/]/.test(value);
  const looksUrl = /^https?:\/\//i.test(value);
  const looksRepo = !looksLocal && !looksUrl && HF_REPO_PATTERN.test(value);
  const isInstallable = looksLocal || looksUrl || looksRepo;
  const localKind = looksLocal ? classifyLocalPath(value) : null;

  return {
    isInstallable,
    label: looksLocal ? `${localKind} path` : looksUrl ? 'URL' : looksRepo ? 'Hugging Face repo' : null,
    localKind,
    looksLocal,
    looksRepo,
    looksUrl,
  };
};
