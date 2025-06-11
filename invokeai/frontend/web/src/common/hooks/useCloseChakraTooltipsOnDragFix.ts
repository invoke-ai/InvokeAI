import { useEffect } from 'react';

// Chakra tooltips sometimes open during a drag operation. We can fix it by dispatching an event that chakra listens
// for to close tooltips. It's reaching into the internals but it seems to work.

const closeEventName = 'chakra-ui:close-tooltip';

export const useCloseChakraTooltipsOnDragFix = () => {
  useEffect(() => {
    const closeTooltips = () => {
      document.dispatchEvent(new window.CustomEvent(closeEventName));
    };
    document.addEventListener('drag', closeTooltips);

    return () => {
      document.removeEventListener('drag', closeTooltips);
    };
  }, []);
};
