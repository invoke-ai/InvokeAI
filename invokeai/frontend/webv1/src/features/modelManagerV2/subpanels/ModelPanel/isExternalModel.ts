/**
 * Checks if a model path is absolute (external model) or relative (Invoke-controlled).
 * External models have absolute paths like "X:/ModelPath/model.safetensors" or "/home/user/models/model.safetensors".
 * Invoke-controlled models have relative paths like "uuid/model.safetensors".
 */
export const isExternalModel = (path: string): boolean => {
  // Unix absolute path
  if (path.startsWith('/')) {
    return true;
  }
  // Windows absolute path (e.g., "X:/..." or "X:\...")
  if (path.length > 1 && path[1] === ':') {
    return true;
  }
  // Windows UNC path (e.g., "\\ServerName\ShareName\..." or "//ServerName/ShareName/...")
  if (path.startsWith('\\\\') || path.startsWith('//')) {
    return true;
  }
  return false;
};
