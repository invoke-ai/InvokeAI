const HF_REPO_ID = /^[A-Za-z0-9][\w.-]*\/[\w.-]+$/;

/**
 * Whether an installed model's source is the Hugging Face repo `repoId`: the repo itself (also `repo::subfolder` or a
 * huggingface.co URL), or a local folder named like the repo, as an in-place install of a downloaded copy is.
 * Case-insensitive, as Hugging Face repo ids are.
 */
export const isModelFromRepo = (source: string | null | undefined, repoId: string): boolean => {
  if (!source) {
    return false;
  }
  const wanted = repoId.toLowerCase();
  const withoutSubfolder = source.split('::')[0]!.trim();
  const hfPath = /^https?:\/\/(?:www\.)?huggingface\.co\/([^?#]+)/i.exec(withoutSubfolder)?.[1];
  const candidate = (hfPath ? hfPath.split('/').slice(0, 2).join('/') : withoutSubfolder).replace(/[/\\]+$/, '');
  if (HF_REPO_ID.test(candidate)) {
    return candidate.toLowerCase() === wanted;
  }
  const folder = candidate.split(/[/\\]/).pop() ?? '';
  return folder.toLowerCase() === wanted.split('/').pop();
};
