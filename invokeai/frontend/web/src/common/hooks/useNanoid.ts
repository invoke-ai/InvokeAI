import { getPrefixedId, nanoid } from 'features/controlLayers/konva/util';
import { useMemo } from 'react';

export const useNanoid = (prefix?: string) => {
  const id = useMemo(() => {
    if (prefix) {
      return getPrefixedId(prefix);
    } else {
      return nanoid();
    }
  }, [prefix]);

  return id;
};
