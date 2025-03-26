import {
  ExternalLink,
  Flex,
  Grid,
  GridItem,
  Heading,
  Image,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Text,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { deepClone } from 'common/util/deepClone';
import DataViewer from 'features/gallery/components/ImageMetadataViewer/DataViewer';
import { discordLink, githubLink, websiteLink } from 'features/system/store/constants';
import InvokeLogoYellow from 'public/assets/images/invoke-tag-lrg.svg';
import type { ReactElement } from 'react';
import { cloneElement, memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetAppDepsQuery, useGetAppVersionQuery, useGetRuntimeConfigQuery } from 'services/api/endpoints/appInfo';

type AboutModalProps = {
  /* The button to open the Settings Modal */
  children: ReactElement;
};

const AboutModal = ({ children }: AboutModalProps) => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const { t } = useTranslation();
  const { data: runtimeConfig } = useGetRuntimeConfigQuery();
  const { data: dependencies } = useGetAppDepsQuery();
  const { data: appVersion } = useGetAppVersionQuery();

  const localData = useMemo(() => {
    const clonedRuntimeConfig = deepClone(runtimeConfig);

    if (clonedRuntimeConfig && clonedRuntimeConfig.config.remote_api_tokens) {
      clonedRuntimeConfig.config.remote_api_tokens.forEach((remote_api_token) => {
        remote_api_token.token = 'REDACTED';
      });
    }

    return {
      version: appVersion?.version,
      dependencies,
      config: clonedRuntimeConfig?.config,
      set_config_fields: clonedRuntimeConfig?.set_fields,
    };
  }, [appVersion, dependencies, runtimeConfig]);

  return (
    <>
      {cloneElement(children, {
        onClick: onOpen,
      })}
      <Modal isOpen={isOpen} onClose={onClose} isCentered size="2xl" useInert={false}>
        <ModalOverlay />
        <ModalContent maxH="80vh" h="34rem">
          <ModalHeader>{t('accessibility.about')}</ModalHeader>
          <ModalCloseButton />
          <ModalBody display="flex" flexDir="column" gap={4}>
            <Grid templateColumns="repeat(2, 1fr)" h="full">
              <GridItem backgroundColor="base.750" borderRadius="base" p="4" h="full">
                <DataViewer label={t('common.systemInformation')} data={localData} />
              </GridItem>
              <GridItem>
                <Flex flexDir="column" gap={3} justifyContent="center" alignItems="center" h="full">
                  <Image src={InvokeLogoYellow} alt="invoke-logo" w="120px" />
                  {appVersion && <Text>{`v${appVersion?.version}`}</Text>}
                  <Grid templateColumns="repeat(2, 1fr)" gap="3">
                    <GridItem>
                      <ExternalLink href={githubLink} label={t('common.githubLabel')} />
                    </GridItem>
                    <GridItem>
                      <ExternalLink href={discordLink} label={t('common.discordLabel')} />
                    </GridItem>
                  </Grid>
                  <Heading fontSize="large">{t('common.aboutHeading')}</Heading>
                  <Text fontSize="sm">{t('common.aboutDesc')}</Text>
                  <ExternalLink href={websiteLink} label={websiteLink} />
                </Flex>
              </GridItem>
            </Grid>
          </ModalBody>
          <ModalFooter />
        </ModalContent>
      </Modal>
    </>
  );
};

export default memo(AboutModal);
