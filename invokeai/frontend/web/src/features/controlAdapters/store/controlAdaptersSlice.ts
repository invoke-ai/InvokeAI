import {
  PayloadAction,
  Update,
  createEntityAdapter,
  createSlice,
} from '@reduxjs/toolkit';
import {
  ControlNetModelParam,
  IPAdapterModelParam,
  T2IAdapterModelParam,
} from 'features/parameters/types/parameterSchemas';
import { cloneDeep, merge, uniq } from 'lodash-es';
import { appSocketInvocationError } from 'services/events/actions';
import { v4 as uuidv4 } from 'uuid';
import { buildControlAdapter } from '../util/buildControlAdapter';
import { controlAdapterImageProcessed } from './actions';
import {
  CONTROLNET_MODEL_DEFAULT_PROCESSORS as CONTROLADAPTER_MODEL_DEFAULT_PROCESSORS,
  CONTROLNET_PROCESSORS,
} from './constants';
import {
  ControlAdapterConfig,
  ControlAdapterProcessorType,
  ControlAdapterType,
  ControlAdaptersState,
  ControlMode,
  ControlNetConfig,
  RequiredControlAdapterProcessorNode,
  ResizeMode,
  T2IAdapterConfig,
  isControlNet,
  isControlNetOrT2IAdapter,
  isIPAdapter,
  isT2IAdapter,
} from './types';

export const caAdapter = createEntityAdapter<ControlAdapterConfig>();

export const {
  selectById: selectControlAdapterById,
  selectAll: selectControlAdapterAll,
  selectEntities: selectControlAdapterEntities,
  selectIds: selectControlAdapterIds,
  selectTotal: selectControlAdapterTotal,
} = caAdapter.getSelectors();

export const initialControlAdapterState: ControlAdaptersState =
  caAdapter.getInitialState<{
    pendingControlImages: string[];
  }>({
    pendingControlImages: [],
  });

export const selectAllControlNets = (controlAdapters: ControlAdaptersState) =>
  selectControlAdapterAll(controlAdapters).filter(isControlNet);

export const selectValidControlNets = (controlAdapters: ControlAdaptersState) =>
  selectControlAdapterAll(controlAdapters)
    .filter(isControlNet)
    .filter(
      (ca) =>
        ca.isEnabled &&
        ca.model &&
        (Boolean(ca.processedControlImage) ||
          (ca.processorType === 'none' && Boolean(ca.controlImage)))
    );

export const selectAllIPAdapters = (controlAdapters: ControlAdaptersState) =>
  selectControlAdapterAll(controlAdapters).filter(isIPAdapter);

export const selectValidIPAdapters = (controlAdapters: ControlAdaptersState) =>
  selectControlAdapterAll(controlAdapters)
    .filter(isIPAdapter)
    .filter((ca) => ca.isEnabled && ca.model && Boolean(ca.controlImage));

export const selectAllT2IAdapters = (controlAdapters: ControlAdaptersState) =>
  selectControlAdapterAll(controlAdapters).filter(isT2IAdapter);

export const selectValidT2IAdapters = (controlAdapters: ControlAdaptersState) =>
  selectControlAdapterAll(controlAdapters)
    .filter(isT2IAdapter)
    .filter(
      (ca) =>
        ca.isEnabled &&
        ca.model &&
        (Boolean(ca.processedControlImage) ||
          (ca.processorType === 'none' && Boolean(ca.controlImage)))
    );

// TODO: I think we can safely remove this?
// const disableAllIPAdapters = (
//   state: ControlAdaptersState,
//   exclude?: string
// ) => {
//   const updates: Update<ControlAdapterConfig>[] = selectAllIPAdapters(state)
//     .filter((ca) => ca.id !== exclude)
//     .map((ca) => ({
//       id: ca.id,
//       changes: { isEnabled: false },
//     }));
//   caAdapter.updateMany(state, updates);
// };

const disableAllControlNets = (
  state: ControlAdaptersState,
  exclude?: string
) => {
  const updates: Update<ControlAdapterConfig>[] = selectAllControlNets(state)
    .filter((ca) => ca.id !== exclude)
    .map((ca) => ({
      id: ca.id,
      changes: { isEnabled: false },
    }));
  caAdapter.updateMany(state, updates);
};

const disableAllT2IAdapters = (
  state: ControlAdaptersState,
  exclude?: string
) => {
  const updates: Update<ControlAdapterConfig>[] = selectAllT2IAdapters(state)
    .filter((ca) => ca.id !== exclude)
    .map((ca) => ({
      id: ca.id,
      changes: { isEnabled: false },
    }));
  caAdapter.updateMany(state, updates);
};

const disableIncompatibleControlAdapters = (
  state: ControlAdaptersState,
  type: ControlAdapterType,
  exclude?: string
) => {
  if (type === 'controlnet') {
    // we cannot do controlnet + t2i adapter, if we are enabled a controlnet, disable all t2is
    disableAllT2IAdapters(state, exclude);
  }
  if (type === 't2i_adapter') {
    // we cannot do controlnet + t2i adapter, if we are enabled a t2i, disable controlnets
    disableAllControlNets(state, exclude);
  }
};

export const controlAdaptersSlice = createSlice({
  name: 'controlAdapters',
  initialState: initialControlAdapterState,
  reducers: {
    controlAdapterAdded: {
      reducer: (
        state,
        action: PayloadAction<{
          id: string;
          type: ControlAdapterType;
          overrides?: Partial<ControlAdapterConfig>;
        }>
      ) => {
        const { id, type, overrides } = action.payload;
        caAdapter.addOne(state, buildControlAdapter(id, type, overrides));
        disableIncompatibleControlAdapters(state, type, id);
      },
      prepare: ({
        type,
        overrides,
      }: {
        type: ControlAdapterType;
        overrides?: Partial<ControlAdapterConfig>;
      }) => {
        return { payload: { id: uuidv4(), type, overrides } };
      },
    },
    controlAdapterRecalled: (
      state,
      action: PayloadAction<ControlAdapterConfig>
    ) => {
      caAdapter.addOne(state, action.payload);
      const { type, id } = action.payload;
      disableIncompatibleControlAdapters(state, type, id);
    },
    controlAdapterDuplicated: {
      reducer: (
        state,
        action: PayloadAction<{
          id: string;
          newId: string;
        }>
      ) => {
        const { id, newId } = action.payload;
        const controlAdapter = selectControlAdapterById(state, id);
        if (!controlAdapter) {
          return;
        }
        const newControlAdapter = merge(cloneDeep(controlAdapter), {
          id: newId,
          isEnabled: true,
        });
        caAdapter.addOne(state, newControlAdapter);
        const { type } = newControlAdapter;
        disableIncompatibleControlAdapters(state, type, newId);
      },
      prepare: (id: string) => {
        return { payload: { id, newId: uuidv4() } };
      },
    },
    controlAdapterAddedFromImage: {
      reducer: (
        state,
        action: PayloadAction<{
          id: string;
          type: ControlAdapterType;
          controlImage: string;
        }>
      ) => {
        const { id, type, controlImage } = action.payload;
        caAdapter.addOne(
          state,
          buildControlAdapter(id, type, { controlImage })
        );
        disableIncompatibleControlAdapters(state, type, id);
      },
      prepare: (payload: {
        type: ControlAdapterType;
        controlImage: string;
      }) => {
        return { payload: { ...payload, id: uuidv4() } };
      },
    },
    controlAdapterRemoved: (state, action: PayloadAction<{ id: string }>) => {
      caAdapter.removeOne(state, action.payload.id);
    },
    controlAdapterIsEnabledChanged: (
      state,
      action: PayloadAction<{ id: string; isEnabled: boolean }>
    ) => {
      const { id, isEnabled } = action.payload;
      caAdapter.updateOne(state, { id, changes: { isEnabled } });
      if (isEnabled) {
        // we are enabling a control adapter. due to limitations in the current system, we may need to disable other adapters
        // TODO: disable when multiple IP adapters are supported
        const ca = selectControlAdapterById(state, id);
        ca && disableIncompatibleControlAdapters(state, ca.type, id);
      }
    },
    controlAdapterImageChanged: (
      state,
      action: PayloadAction<{
        id: string;
        controlImage: string | null;
      }>
    ) => {
      const { id, controlImage } = action.payload;
      const ca = selectControlAdapterById(state, id);
      if (!ca) {
        return;
      }

      caAdapter.updateOne(state, {
        id,
        changes: { controlImage, processedControlImage: null },
      });

      if (
        controlImage !== null &&
        isControlNetOrT2IAdapter(ca) &&
        ca.processorType !== 'none'
      ) {
        state.pendingControlImages.push(id);
      }
    },
    controlAdapterProcessedImageChanged: (
      state,
      action: PayloadAction<{
        id: string;
        processedControlImage: string | null;
      }>
    ) => {
      const { id, processedControlImage } = action.payload;
      const cn = selectControlAdapterById(state, id);
      if (!cn) {
        return;
      }

      if (!isControlNetOrT2IAdapter(cn)) {
        return;
      }

      caAdapter.updateOne(state, {
        id,
        changes: {
          processedControlImage,
        },
      });

      state.pendingControlImages = state.pendingControlImages.filter(
        (pendingId) => pendingId !== id
      );
    },
    controlAdapterModelCleared: (
      state,
      action: PayloadAction<{ id: string }>
    ) => {
      caAdapter.updateOne(state, {
        id: action.payload.id,
        changes: { model: null },
      });
    },
    controlAdapterModelChanged: (
      state,
      action: PayloadAction<{
        id: string;
        model:
          | ControlNetModelParam
          | T2IAdapterModelParam
          | IPAdapterModelParam;
      }>
    ) => {
      const { id, model } = action.payload;
      const cn = selectControlAdapterById(state, id);
      if (!cn) {
        return;
      }

      if (!isControlNetOrT2IAdapter(cn)) {
        caAdapter.updateOne(state, { id, changes: { model } });
        return;
      }

      const update: Update<ControlNetConfig | T2IAdapterConfig> = {
        id,
        changes: { model },
      };

      update.changes.processedControlImage = null;

      if (cn.shouldAutoConfig) {
        let processorType: ControlAdapterProcessorType | undefined = undefined;

        for (const modelSubstring in CONTROLADAPTER_MODEL_DEFAULT_PROCESSORS) {
          if (model.model_name.includes(modelSubstring)) {
            processorType =
              CONTROLADAPTER_MODEL_DEFAULT_PROCESSORS[modelSubstring];
            break;
          }
        }

        if (processorType) {
          update.changes.processorType = processorType;
          update.changes.processorNode = CONTROLNET_PROCESSORS[processorType]
            .default as RequiredControlAdapterProcessorNode;
        } else {
          update.changes.processorType = 'none';
          update.changes.processorNode = CONTROLNET_PROCESSORS.none
            .default as RequiredControlAdapterProcessorNode;
        }
      }

      caAdapter.updateOne(state, update);
    },
    controlAdapterWeightChanged: (
      state,
      action: PayloadAction<{ id: string; weight: number }>
    ) => {
      const { id, weight } = action.payload;
      caAdapter.updateOne(state, { id, changes: { weight } });
    },
    controlAdapterBeginStepPctChanged: (
      state,
      action: PayloadAction<{ id: string; beginStepPct: number }>
    ) => {
      const { id, beginStepPct } = action.payload;
      caAdapter.updateOne(state, { id, changes: { beginStepPct } });
    },
    controlAdapterEndStepPctChanged: (
      state,
      action: PayloadAction<{ id: string; endStepPct: number }>
    ) => {
      const { id, endStepPct } = action.payload;
      caAdapter.updateOne(state, { id, changes: { endStepPct } });
    },
    controlAdapterControlModeChanged: (
      state,
      action: PayloadAction<{ id: string; controlMode: ControlMode }>
    ) => {
      const { id, controlMode } = action.payload;
      const cn = selectControlAdapterById(state, id);
      if (!cn || !isControlNet(cn)) {
        return;
      }
      caAdapter.updateOne(state, { id, changes: { controlMode } });
    },
    controlAdapterResizeModeChanged: (
      state,
      action: PayloadAction<{
        id: string;
        resizeMode: ResizeMode;
      }>
    ) => {
      const { id, resizeMode } = action.payload;
      const cn = selectControlAdapterById(state, id);
      if (!cn || !isControlNetOrT2IAdapter(cn)) {
        return;
      }
      caAdapter.updateOne(state, { id, changes: { resizeMode } });
    },
    controlAdapterProcessorParamsChanged: (
      state,
      action: PayloadAction<{
        id: string;
        params: Partial<RequiredControlAdapterProcessorNode>;
      }>
    ) => {
      const { id, params } = action.payload;
      const cn = selectControlAdapterById(state, id);
      if (!cn || !isControlNetOrT2IAdapter(cn) || !cn.processorNode) {
        return;
      }

      const processorNode = merge(cloneDeep(cn.processorNode), params);

      caAdapter.updateOne(state, {
        id,
        changes: {
          shouldAutoConfig: false,
          processorNode,
        },
      });
    },
    controlAdapterProcessortTypeChanged: (
      state,
      action: PayloadAction<{
        id: string;
        processorType: ControlAdapterProcessorType;
      }>
    ) => {
      const { id, processorType } = action.payload;
      const cn = selectControlAdapterById(state, id);
      if (!cn || !isControlNetOrT2IAdapter(cn)) {
        return;
      }

      const processorNode = cloneDeep(
        CONTROLNET_PROCESSORS[processorType].default
      ) as RequiredControlAdapterProcessorNode;

      caAdapter.updateOne(state, {
        id,
        changes: {
          processorType,
          processedControlImage: null,
          processorNode,
          shouldAutoConfig: false,
        },
      });
    },
    controlAdapterAutoConfigToggled: (
      state,
      action: PayloadAction<{
        id: string;
      }>
    ) => {
      const { id } = action.payload;
      const cn = selectControlAdapterById(state, id);
      if (!cn || !isControlNetOrT2IAdapter(cn)) {
        return;
      }

      const update: Update<ControlNetConfig | T2IAdapterConfig> = {
        id,
        changes: { shouldAutoConfig: !cn.shouldAutoConfig },
      };

      if (update.changes.shouldAutoConfig) {
        // manage the processor for the user
        let processorType: ControlAdapterProcessorType | undefined = undefined;

        for (const modelSubstring in CONTROLADAPTER_MODEL_DEFAULT_PROCESSORS) {
          if (cn.model?.model_name.includes(modelSubstring)) {
            processorType =
              CONTROLADAPTER_MODEL_DEFAULT_PROCESSORS[modelSubstring];
            break;
          }
        }

        if (processorType) {
          update.changes.processorType = processorType;
          update.changes.processorNode = CONTROLNET_PROCESSORS[processorType]
            .default as RequiredControlAdapterProcessorNode;
        } else {
          update.changes.processorType = 'none';
          update.changes.processorNode = CONTROLNET_PROCESSORS.none
            .default as RequiredControlAdapterProcessorNode;
        }
      }

      caAdapter.updateOne(state, update);
    },
    controlAdaptersReset: () => {
      return cloneDeep(initialControlAdapterState);
    },
    pendingControlImagesCleared: (state) => {
      state.pendingControlImages = [];
    },
  },
  extraReducers: (builder) => {
    builder.addCase(controlAdapterImageProcessed, (state, action) => {
      const cn = selectControlAdapterById(state, action.payload.id);
      if (!cn) {
        return;
      }
      if (cn.controlImage !== null) {
        state.pendingControlImages = uniq(
          state.pendingControlImages.concat(action.payload.id)
        );
      }
    });

    builder.addCase(appSocketInvocationError, (state) => {
      state.pendingControlImages = [];
    });
  },
});

export const {
  controlAdapterAdded,
  controlAdapterRecalled,
  controlAdapterDuplicated,
  controlAdapterAddedFromImage,
  controlAdapterRemoved,
  controlAdapterImageChanged,
  controlAdapterProcessedImageChanged,
  controlAdapterIsEnabledChanged,
  controlAdapterModelChanged,
  controlAdapterWeightChanged,
  controlAdapterBeginStepPctChanged,
  controlAdapterEndStepPctChanged,
  controlAdapterControlModeChanged,
  controlAdapterResizeModeChanged,
  controlAdapterProcessorParamsChanged,
  controlAdapterProcessortTypeChanged,
  controlAdaptersReset,
  controlAdapterAutoConfigToggled,
  pendingControlImagesCleared,
  controlAdapterModelCleared,
} = controlAdaptersSlice.actions;

export default controlAdaptersSlice.reducer;
