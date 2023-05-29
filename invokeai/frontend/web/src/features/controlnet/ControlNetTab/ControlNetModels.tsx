import { ActionCreatorWithPayload } from '@reduxjs/toolkit';
import { useAppDispatch } from 'app/store/storeHooks';
import IAICustomSelect from 'common/components/IAICustomSelect';
import { useEffect, useState } from 'react';
import { ModelsService } from 'services/api';
import { ControlNetConfig } from '../store/controlnetTypes';

interface ControlNetModelsProps {
  controlnet: ControlNetConfig;
  setControlnet: ActionCreatorWithPayload<ControlNetConfig>;
}

export default function ControlNetModels(props: ControlNetModelsProps) {
  const { controlnet, setControlnet } = props;

  const [controlnetModels, setControlNetModels] = useState<string[] | null>(
    null
  );

  const dispatch = useAppDispatch();

  const getControlNetModels = async () => {
    const response = await ModelsService.listControlnetModels();
    setControlNetModels(Object.keys(response['controlnet_models']));
  };

  useEffect(() => {
    getControlNetModels();
  }, []);

  const handleModelSelect = (v: string | null | undefined) => {
    dispatch(
      setControlnet({
        ...controlnet,
        controlnetModel: v,
      })
    );
  };

  return (
    <IAICustomSelect
      label="Model"
      items={controlnetModels ? controlnetModels : ['']}
      selectedItem={
        controlnet.controlnetModel ? controlnet.controlnetModel : 'none'
      }
      setSelectedItem={handleModelSelect}
    />
  );
}
