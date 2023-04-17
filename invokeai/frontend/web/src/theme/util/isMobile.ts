export let isMobile = false;

export function updateIsMobile() {
  isMobile = window.innerWidth <= 768 || document.body.clientWidth <= 768;
}
