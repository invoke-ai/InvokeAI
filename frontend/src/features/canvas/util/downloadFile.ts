/**
 * Downloads a file, given its URL.
 */
const downloadFile = (url: string) => {
  const a = document.createElement('a');
  a.href = url;
  a.download = '';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  a.remove();
};

export default downloadFile;
