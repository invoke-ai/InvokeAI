type Base64AndCaption = {
  base64: string;
  caption: string;
};

const openBase64ImageInTab = (images: Base64AndCaption[]) => {
  const w = window.open('');
  if (!w) {
    return;
  }

  images.forEach((i) => {
    const image = new Image();
    image.src = i.base64;

    w.document.write(i.caption);
    w.document.write('</br>');
    w.document.write(image.outerHTML);
    w.document.write('</br></br>');
  });
};

export default openBase64ImageInTab;
