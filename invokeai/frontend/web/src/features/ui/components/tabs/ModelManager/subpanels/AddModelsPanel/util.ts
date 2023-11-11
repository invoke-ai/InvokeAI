export function getModelName(filepath: string, isCheckpoint: boolean = true) {
  let regex;
  if (isCheckpoint) {
    regex = new RegExp('[^\\\\/]+(?=\\.)');
  } else {
    regex = new RegExp('[^\\\\/]+(?=[\\\\/]?$)');
  }

  const match = filepath.match(regex);
  if (match) {
    return match[0];
  } else {
    return '';
  }
}
