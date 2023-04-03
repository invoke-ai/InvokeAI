import { Box } from '@chakra-ui/react';
import { getTextualInversionTriggers } from 'app/socketio/actions';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISimpleMenu, { IAIMenuItem } from 'common/components/IAISimpleMenu';
import { setPrompt } from 'features/parameters/store/generationSlice';
import { useEffect, useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export default function TextualInversionManager() {
  const dispatch = useAppDispatch();
  const prompt = useAppSelector((state) => state.generation.prompt);
  const foundTextualInversionTriggers = useAppSelector(
    (state) => state.system.foundTextualInversionTriggers
  );
  const [textualInversionItems, setTextualInversionItems] = useState<
    IAIMenuItem[]
  >([{ item: '', onClick: undefined }]);
  const { t } = useTranslation();

  const textualInversionExists = useCallback(
    (textualInversion: string) => {
      const textualInversion_regex = new RegExp(`${textualInversion}`);
      if (prompt.match(textualInversion_regex)) return true;
      return false;
    },
    [prompt]
  );

  const handleTextualInversion = useCallback(
    (textualInversion: string) => {
      if (textualInversionExists(textualInversion)) {
        const textualInversion_regex = new RegExp(`${textualInversion}`);
        const newPrompt = prompt.replace(textualInversion_regex, '');
        dispatch(setPrompt(newPrompt));
      } else {
        dispatch(setPrompt(`${prompt} ${textualInversion}`));
      }
    },
    [dispatch, textualInversionExists, prompt]
  );

  useEffect(() => {
    dispatch(getTextualInversionTriggers());
  }, [dispatch]);

  const renderTextualInversionOption = useCallback(
    (textualInversion: string) => {
      const thisTextualInversionExists =
        textualInversionExists(textualInversion);
      const textualInversionExistsStyle = {
        fontWeight: 'bold',
        color: 'var(--context-menu-active-item)',
      };
      return (
        <Box
          style={thisTextualInversionExists ? textualInversionExistsStyle : {}}
        >
          {textualInversion}
        </Box>
      );
    },
    [textualInversionExists]
  );

  useEffect(() => {
    if (foundTextualInversionTriggers) {
      console.log('rendertextualinversionoption: here i am');
      const textualInversionsFound: IAIMenuItem[] = [];
      foundTextualInversionTriggers.forEach((textualInversion) => {
        if (textualInversion.name !== ' ') {
          const newTextualInversionItem: IAIMenuItem = {
            item: renderTextualInversionOption(textualInversion.name),
            onClick: () => handleTextualInversion(textualInversion.name),
          };
          textualInversionsFound.push(newTextualInversionItem);
        }
      });
      setTextualInversionItems(textualInversionsFound);
    }
  }, [
    foundTextualInversionTriggers,
    handleTextualInversion,
    renderTextualInversionOption,
  ]);

  return foundTextualInversionTriggers &&
    foundTextualInversionTriggers?.length > 0 ? (
    <IAISimpleMenu
      menuItems={textualInversionItems}
      menuType="regular"
      buttonText={t('modelManager.addTextualInversionTrigger')}
    />
  ) : (
    <Box
      background="var(--btn-base-color)"
      padding={2}
      textAlign="center"
      borderRadius={4}
      fontWeight="bold"
    >
      {t('modelManager.noTextualInversionTriggers')}
    </Box>
  );
}
