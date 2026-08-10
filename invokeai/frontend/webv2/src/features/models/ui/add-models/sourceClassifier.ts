/** owner/repo with optional :variant[:path] qualifiers - the HF repo shape. */
const HF_REPO_PATTERN = /^[\w.-]+\/[\w.-]+(:[\w./-]*)*$/;
// 12 covers the longest real model extension, ".safetensors" (11 letters).
const FILE_EXTENSION_PATTERN = /\.[A-Za-z0-9]{1,12}$/;
/**
 * Extensions that mark a relative path as a model file rather than an HF repo
 * id. Deliberately narrower than FILE_EXTENSION_PATTERN: real repo ids end in
 * dot-suffixes too ("stabilityai/stable-diffusion-2.1").
 */
const MODEL_FILE_EXTENSION_PATTERN = /\.(safetensors|ckpt|pt|pth|bin|gguf|onnx|pkl)$/i;
const RELATIVE_PATH_PATTERN = /^\.{1,2}[\\/]/;

export interface SourceKind {
  looksLocal: boolean;
  looksUrl: boolean;
  looksRepo: boolean;
  isInstallable: boolean;
  localKind: 'file' | 'folder' | null;
  /** i18n key for the detected source kind, or null when it reads as a search. */
  labelKey: string | null;
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
  // "models/foo.safetensors" and "./repo" satisfy the HF repo shape but are
  // clearly path-like; they fall through to search, since a relative path is
  // not an installable source. The extension check runs on the pre-colon
  // portion so "owner/repo:fp16:path/file.safetensors" stays a repo.
  const looksRelativePath =
    RELATIVE_PATH_PATTERN.test(value) || MODEL_FILE_EXTENSION_PATTERN.test(value.split(':', 1)[0] ?? '');
  const looksRepo = !looksLocal && !looksUrl && !looksRelativePath && HF_REPO_PATTERN.test(value);
  const isInstallable = looksLocal || looksUrl || looksRepo;
  const localKind = looksLocal ? classifyLocalPath(value) : null;

  return {
    isInstallable,
    labelKey: looksLocal
      ? localKind === 'file'
        ? 'models.sourceKind.filePath'
        : 'models.sourceKind.folderPath'
      : looksUrl
        ? 'models.sourceKind.url'
        : looksRepo
          ? 'models.sourceKind.hfRepo'
          : null,
    localKind,
    looksLocal,
    looksRepo,
    looksUrl,
  };
};
