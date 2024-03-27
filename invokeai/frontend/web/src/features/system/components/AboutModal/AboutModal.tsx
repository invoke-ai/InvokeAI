import {
  ExternalLink,
  Flex,
  Grid,
  GridItem,
  Heading,
  IconButton,
  Image,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Spacer,
  Text,
  Tooltip,
  useDisclosure,
} from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { discordLink, githubLink, websiteLink } from 'features/system/store/constants';
import { map } from 'lodash-es';
/** @knipignore */
import InvokeLogoYellow from 'public/assets/images/invoke-tag-lrg.svg';
import type { ReactElement } from 'react';
import { cloneElement, memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyBold } from 'react-icons/pi';
import { useGetAppDepsQuery, useGetAppVersionQuery } from 'services/api/endpoints/appInfo';

type AboutModalProps = {
  /* The button to open the Settings Modal */
  children: ReactElement;
};

const AboutModal = ({ children }: AboutModalProps) => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const { t } = useTranslation();
  const { depsArray, depsObject } = useGetAppDepsQuery(undefined, {
    selectFromResult: ({ data }) => ({
      depsObject: data,
      depsArray: data ? map(data, (version, name) => ({ name, version })) : [],
    }),
  });
  const { data: appVersion } = useGetAppVersionQuery();

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(JSON.stringify(depsObject, null, 2));
  }, [depsObject]);

  return (
    <>
      {cloneElement(children, {
        onClick: onOpen,
      })}
      <Modal isOpen={isOpen} onClose={onClose} isCentered size="2xl">
        <ModalOverlay />
        <ModalContent maxH="80vh" h="34rem">
          <ModalHeader>{t('accessibility.about')}</ModalHeader>
          <ModalCloseButton />
          <ModalBody display="flex" flexDir="column" gap={4}>
            <Grid templateColumns="repeat(2, 1fr)" h="full">
              <GridItem backgroundColor="base.750" borderRadius="base" p="4" h="full">
                <ScrollableContent>
                  <Flex position="sticky" top="0" backgroundColor="base.750" p={1} alignItems="center">
                    <Heading size="md">{t('common.localSystem')}</Heading>
                    <Spacer />
                    <Tooltip label={t('common.copy')}>
                      <IconButton
                        onClick={handleCopy}
                        isDisabled={!depsObject}
                        aria-label={t('common.copy')}
                        icon={<PiCopyBold />}
                        variant="ghost"
                      />
                    </Tooltip>
                  </Flex>
                  {depsArray.map(({ name, version }, i) => (
                    <Grid key={i} py="2" px="1" w="full" templateColumns="repeat(2, 1fr)">
                      <Text>{name}</Text>
                      <Text>{version ? version : t('common.notInstalled')}</Text>
                    </Grid>
                  ))}
                </ScrollableContent>
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
