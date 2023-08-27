import { Box, Flex } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import ProcessButtons from 'features/parameters/components/ProcessButtons/ProcessButtons';
import SDXLImageToImageTabParameters from 'features/sdxl/components/SDXLImageToImageTabParameters';
import SDXLTextToImageTabParameters from 'features/sdxl/components/SDXLTextToImageTabParameters';
import SDXLUnifiedCanvasTabParameters from 'features/sdxl/components/SDXLUnifiedCanvasTabParameters';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import { PropsWithChildren, memo } from 'react';
import { activeTabNameSelector } from '../store/uiSelectors';
import ImageToImageTabParameters from './tabs/ImageToImage/ImageToImageTabParameters';
import TextToImageTabParameters from './tabs/TextToImage/TextToImageTabParameters';
import UnifiedCanvasParameters from './tabs/UnifiedCanvas/UnifiedCanvasParameters';

const ParametersPanel = () => {
  const activeTabName = useAppSelector(activeTabNameSelector);
  const model = useAppSelector((state: RootState) => state.generation.model);

  if (activeTabName === 'txt2img') {
    return (
      <ParametersPanelWrapper>
        {model && model.base_model === 'sdxl' ? (
          <SDXLTextToImageTabParameters />
        ) : (
          <TextToImageTabParameters />
        )}
      </ParametersPanelWrapper>
    );
  }

  if (activeTabName === 'img2img') {
    return (
      <ParametersPanelWrapper>
        {model && model.base_model === 'sdxl' ? (
          <SDXLImageToImageTabParameters />
        ) : (
          <ImageToImageTabParameters />
        )}
      </ParametersPanelWrapper>
    );
  }

  if (activeTabName === 'unifiedCanvas') {
    return (
      <ParametersPanelWrapper>
        {model && model.base_model === 'sdxl' ? (
          <SDXLUnifiedCanvasTabParameters />
        ) : (
          <UnifiedCanvasParameters />
        )}
      </ParametersPanelWrapper>
    );
  }

  return null;
};

export default memo(ParametersPanel);

const ParametersPanelWrapper = memo((props: PropsWithChildren) => {
  return (
    <Flex
      sx={{
        w: 'full',
        h: 'full',
        flexDir: 'column',
        gap: 2,
      }}
    >
      <ProcessButtons />
      <Flex
        layerStyle="first"
        sx={{
          w: 'full',
          h: 'full',
          position: 'relative',
          borderRadius: 'base',
          p: 2,
        }}
      >
        <Flex
          sx={{
            w: 'full',
            h: 'full',
            position: 'relative',
          }}
        >
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
            }}
          >
            <OverlayScrollbarsComponent
              defer
              style={{ height: '100%', width: '100%' }}
              options={{
                scrollbars: {
                  visibility: 'auto',
                  autoHide: 'scroll',
                  autoHideDelay: 800,
                  theme: 'os-theme-dark',
                },
                overflow: {
                  x: 'hidden',
                },
              }}
            >
              <Flex
                sx={{
                  gap: 2,
                  flexDirection: 'column',
                  h: 'full',
                  w: 'full',
                }}
              >
                {props.children}
              </Flex>
            </OverlayScrollbarsComponent>
          </Box>
        </Flex>
      </Flex>
    </Flex>
  );
});

ParametersPanelWrapper.displayName = 'ParametersPanelWrapper';
