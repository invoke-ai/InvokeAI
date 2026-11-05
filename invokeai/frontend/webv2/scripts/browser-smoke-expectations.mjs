/** WebKit denies these old-document requests before dispatch during the deliberate reload recovery scenario. */
export const isExpectedWebKitUnloadError = (event, { engine, navigating, origin }) => {
  if (engine !== 'webkit' || !navigating || event.kind !== 'pageerror') {
    return false;
  }
  const match = /^(Fetch API|XMLHttpRequest) cannot load (https?:\/\/\S+) due to access control checks\.$/.exec(
    event.message.split('\n', 1)[0]
  );
  if (!match) {
    return false;
  }
  const url = new URL(match[2]);
  return (
    url.origin === origin &&
    ((match[1] === 'Fetch API' && url.pathname === '/api/v1/projects/fixture-project-002/workspace' && !url.search) ||
      (match[1] === 'XMLHttpRequest' &&
        url.pathname === '/ws/socket.io/' &&
        url.searchParams.get('transport') === 'polling'))
  );
};
