import { Flex, FormControl, FormLabel, IconButton, Image, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import ProgressImage from 'features/gallery/components/CurrentImage/ProgressImage';
import { setShowSeamless } from 'features/showcase/store/showcaseSlice';
import { setShouldShowShowcase } from 'features/ui/store/uiSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

import SeamlessTextureViewer from './SeamlessTextureViewer';

type ShowcaseProps = {
  imageDTO: ImageDTO;
};

function Showcase(props: ShowcaseProps) {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { imageDTO } = props;
  const showSeamless = useAppSelector((s) => s.showcase.showSeamless);
  const shouldShowProgressInViewer = useAppSelector((s) => s.ui.shouldShowProgressInViewer);
  const hasDenoiseProgress = useAppSelector((s) => Boolean(s.system.denoiseProgress));

  const closeShowcase = useCallback(() => {
    dispatch(setShouldShowShowcase(null));
  }, [dispatch]);

  const handleShowSeamless = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(setShowSeamless(e.target.checked)),
    [dispatch]
  );

  return (
    <Flex
      sx={{
        w: 'full',
        h: 'full',
        position: 'relative',
        background: 'base.850',
        flexDir: 'column',
        gap: 2,
      }}
    >
      <Flex sx={{ background: 'base.900', borderRadius: 'base', p: 2 }}>
        <FormControl sx={{ gap: 0, pl: 2 }}>
          <FormLabel>{t('showcase.seamlessChecker')}</FormLabel>
          <Switch isChecked={showSeamless} onChange={handleShowSeamless} />
        </FormControl>
        <IconButton icon={<PiXBold fontSize={12} />} onClick={closeShowcase} aria-label={t('common.close')} />
      </Flex>
      <Flex
        sx={{
          w: 'full',
          h: '100%',
          justifyContent: 'center',
          background: 'base.900',
          borderRadius: 'base',
          overflow: 'hidden',
        }}
      >
        {showSeamless ? (
          <SeamlessTextureViewer imageDTO={imageDTO} />
        ) : (
          <Flex p={4}>
            <Image src={imageDTO.image_url} w="auto" h="auto" objectFit="contain" />
          </Flex>
        )}
      </Flex>
      {shouldShowProgressInViewer && hasDenoiseProgress && (
        <Flex
          sx={{
            position: 'absolute',
            top: 16,
            left: 0,
            w: '256px',
            h: '256px',
            margin: 2,
            borderRadius: 4,
            overflow: 'clip',
          }}
        >
          <ProgressImage />
        </Flex>
      )}
    </Flex>
  );
}

export default memo(Showcase);
