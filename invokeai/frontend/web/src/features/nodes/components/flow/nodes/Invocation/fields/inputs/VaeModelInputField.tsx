import { SelectItem } from '@mantine/core';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import IAIMantineSelectItemWithTooltip from 'common/components/IAIMantineSelectItemWithTooltip';
import { fieldVaeModelValueChanged } from 'features/nodes/store/nodesSlice';
import {
  FieldComponentProps,
  VaeModelInputFieldTemplate,
  VaeModelInputFieldValue,
} from 'features/nodes/types/types';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { modelIdToVAEModelParam } from 'features/parameters/util/modelIdToVAEModelParam';
import { forEach } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useGetVaeModelsQuery } from 'services/api/endpoints/models';

const VaeModelInputFieldComponent = (
  props: FieldComponentProps<
    VaeModelInputFieldValue,
    VaeModelInputFieldTemplate
  >
) => {
  const { nodeId, field } = props;
  const vae = field.value;
  const dispatch = useAppDispatch();
  const { data: vaeModels } = useGetVaeModelsQuery();

  const data = useMemo(() => {
    if (!vaeModels) {
      return [];
    }

    const data: SelectItem[] = [
      {
        value: 'default',
        label: 'Default',
        group: 'Default',
      },
    ];

    forEach(vaeModels.entities, (vae, id) => {
      if (!vae) {
        return;
      }

      data.push({
        value: id,
        label: vae.model_name,
        group: MODEL_TYPE_MAP[vae.base_model],
      });
    });

    return data.sort((a, b) => (a.disabled && !b.disabled ? 1 : -1));
  }, [vaeModels]);

  // grab the full model entity from the RTK Query cache
  const selectedVaeModel = useMemo(
    () =>
      vaeModels?.entities[`${vae?.base_model}/vae/${vae?.model_name}`] ?? null,
    [vaeModels?.entities, vae]
  );

  const handleChangeModel = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }

      const newVaeModel = modelIdToVAEModelParam(v);

      if (!newVaeModel) {
        return;
      }

      dispatch(
        fieldVaeModelValueChanged({
          nodeId,
          fieldName: field.name,
          value: newVaeModel,
        })
      );
    },
    [dispatch, field.name, nodeId]
  );

  return (
    <IAIMantineSearchableSelect
      className="nowheel nodrag"
      itemComponent={IAIMantineSelectItemWithTooltip}
      tooltip={selectedVaeModel?.description}
      value={selectedVaeModel?.id ?? 'default'}
      placeholder="Default"
      data={data}
      onChange={handleChangeModel}
      disabled={data.length === 0}
      error={!selectedVaeModel}
      clearable
      sx={{
        width: '100%',
        '.mantine-Select-dropdown': {
          width: '16rem !important',
        },
      }}
    />
  );
};

export default memo(VaeModelInputFieldComponent);
