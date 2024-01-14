export const isInputElement = (el: HTMLElement) => {
  return (
    el.tagName.toLowerCase() === 'input' ||
    el.tagName.toLowerCase() === 'textarea' ||
    el.tagName.toLowerCase() === 'select'
  );
};
