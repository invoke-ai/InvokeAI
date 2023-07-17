import { filter } from 'lodash-es';
import { ControlNetConfig } from '../store/controlNetSlice';

export const getValidControlNets = (
  controlNets: Record<string, ControlNetConfig>
) => {
  const validControlNets = filter(
    controlNets,
    (c) =>
      c.isEnabled &&
      (Boolean(c.processedControlImage) ||
        (c.processorType === 'none' && Boolean(c.controlImage)))
  );
  return validControlNets;
};
