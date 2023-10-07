import { isAnyOf } from '@reduxjs/toolkit';
import {
  controlAdapterAdded,
  controlAdapterAddedFromImage,
  controlAdapterIsEnabledChanged,
  controlAdapterRecalled,
  selectControlAdapterAll,
  selectControlAdapterById,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { ControlAdapterType } from 'features/controlAdapters/store/types';
import { addToast } from 'features/system/store/systemSlice';
import i18n from 'i18n';
import { startAppListening } from '..';

const isAnyControlAdapterAddedOrEnabled = isAnyOf(
  controlAdapterAdded,
  controlAdapterAddedFromImage,
  controlAdapterRecalled,
  controlAdapterIsEnabledChanged
);

/**
 * Until we can have both controlnet and t2i adapter enabled at once, they are mutually exclusive
 * This displays a toast when one is enabled and the other is already enabled, or one is added
 * with the other enabled
 */
export const addControlAdapterAddedOrEnabledListener = () => {
  startAppListening({
    matcher: isAnyControlAdapterAddedOrEnabled,
    effect: async (action, { dispatch, getOriginalState }) => {
      const controlAdapters = getOriginalState().controlAdapters;

      const hasEnabledControlNets = selectControlAdapterAll(
        controlAdapters
      ).some((ca) => ca.isEnabled && ca.type === 'controlnet');

      const hasEnabledT2IAdapters = selectControlAdapterAll(
        controlAdapters
      ).some((ca) => ca.isEnabled && ca.type === 't2i_adapter');

      let caType: ControlAdapterType | null = null;

      if (controlAdapterAdded.match(action)) {
        caType = action.payload.type;
      }

      if (controlAdapterAddedFromImage.match(action)) {
        caType = action.payload.type;
      }

      if (controlAdapterRecalled.match(action)) {
        caType = action.payload.type;
      }

      if (controlAdapterIsEnabledChanged.match(action)) {
        const _caType = selectControlAdapterById(
          controlAdapters,
          action.payload.id
        )?.type;
        if (!_caType) {
          return;
        }
        caType = _caType;
      }

      if (
        (caType === 'controlnet' && hasEnabledT2IAdapters) ||
        (caType === 't2i_adapter' && hasEnabledControlNets)
      ) {
        const title =
          caType === 'controlnet'
            ? i18n.t('controlnet.controlNetEnabledT2IDisabled')
            : i18n.t('controlnet.t2iEnabledControlNetDisabled');

        const description = i18n.t('controlnet.controlNetT2IMutexDesc');

        dispatch(
          addToast({
            title,
            description,
            status: 'warning',
          })
        );
      }
    },
  });
};
