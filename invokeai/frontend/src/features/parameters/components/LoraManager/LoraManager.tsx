import { Box } from '@chakra-ui/react';
import { getLoraModels } from 'app/socketio/actions';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISimpleMenu, { IAIMenuItem } from 'common/components/IAISimpleMenu';
import { setPrompt } from 'features/parameters/store/generationSlice';
import { useEffect, useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export default function LoraManager() {
  const dispatch = useAppDispatch();
  const prompt = useAppSelector((state) => state.generation.prompt);
  const foundLoras = useAppSelector((state) => state.system.foundLoras);
  const [loraItems, setLoraItems] = useState<IAIMenuItem[]>([
    { item: '', onClick: undefined },
  ]);
  const { t } = useTranslation();

  const loraExists = useCallback(
    (lora: string) => {
      const lora_regex = new RegExp(`withLora\\(${lora},?\\s*([\\d.]+)?\\)`);
      if (prompt.match(lora_regex)) return true;
      return false;
    },
    [prompt]
  );

  const handleLora = useCallback(
    (lora: string) => {
      if (loraExists(lora)) {
        const lora_regex = new RegExp(`withLora\\(${lora},?\\s*([\\d.]+)?\\)`);
        const newPrompt = prompt.replace(lora_regex, '');
        dispatch(setPrompt(newPrompt));
      } else {
        dispatch(setPrompt(`${prompt} withLora(${lora},1)`));
      }
    },
    [dispatch, loraExists, prompt]
  );

  useEffect(() => {
    dispatch(getLoraModels());
  }, [dispatch]);

  const renderLoraOption = useCallback(
    (lora: string) => {
      const thisloraExists = loraExists(lora);
      const loraExistsStyle = {
        fontWeight: 'bold',
        color: 'var(--context-menu-active-item)',
      };
      return <Box style={thisloraExists ? loraExistsStyle : {}}>{lora}</Box>;
    },
    [loraExists]
  );

  useEffect(() => {
    if (foundLoras) {
      const lorasFound: IAIMenuItem[] = [];
      foundLoras.forEach((lora) => {
        if (lora.name !== ' ') {
          const newLoraItem: IAIMenuItem = {
            item: renderLoraOption(lora.name),
            onClick: () => handleLora(lora.name),
          };
          lorasFound.push(newLoraItem);
        }
      });
      setLoraItems(lorasFound);
    }
  }, [foundLoras, loraItems, dispatch, prompt, handleLora, renderLoraOption]);

  return (
    foundLoras &&
    foundLoras?.length > 0 && (
      <IAISimpleMenu
        menuItems={loraItems}
        menuType="regular"
        buttonText={t('modelManager.addLora')}
      />
    )
  );
}
