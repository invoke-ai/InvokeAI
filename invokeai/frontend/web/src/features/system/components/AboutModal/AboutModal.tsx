import { ExternalLinkIcon } from '@chakra-ui/icons';
import {
  Flex,
  Grid,
  GridItem,
  Image,
  Link,
  useDisclosure,
} from '@chakra-ui/react';
import { InvHeading } from 'common/components/InvHeading/wrapper';
import {
  InvModal,
  InvModalBody,
  InvModalCloseButton,
  InvModalContent,
  InvModalFooter,
  InvModalHeader,
  InvModalOverlay,
} from 'common/components/InvModal/wrapper';
import { InvText } from 'common/components/InvText/wrapper';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import {
  discordLink,
  githubLink,
  websiteLink,
} from 'features/system/store/constants';
import { map } from 'lodash-es';
import InvokeLogoYellow from 'public/assets/images/invoke-tag-lrg.svg';
import type { ReactElement } from 'react';
import { cloneElement, memo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  useGetAppDepsQuery,
  useGetAppVersionQuery,
} from 'services/api/endpoints/appInfo';

type AboutModalProps = {
  /* The button to open the Settings Modal */
  children: ReactElement;
};

const AboutModal = ({ children }: AboutModalProps) => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const { t } = useTranslation();
  const { deps } = useGetAppDepsQuery(undefined, {
    selectFromResult: ({ data }) => ({
      deps: data ? map(data, (version, name) => ({ name, version })) : [],
    }),
  });
  const { data: appVersion } = useGetAppVersionQuery();

  return (
    <>
      {cloneElement(children, {
        onClick: onOpen,
      })}
      <InvModal isOpen={isOpen} onClose={onClose} isCentered size="2xl">
        <InvModalOverlay />
        <InvModalContent maxH="80vh" h="80vh">
          <InvModalHeader>{t('accessibility.about')}</InvModalHeader>
          <InvModalCloseButton />
          <InvModalBody display="flex" flexDir="column" gap={4}>
            <Grid templateColumns="repeat(2, 1fr)" h="full">
              <GridItem
                backgroundColor="base.750"
                borderRadius="base"
                p="4"
                h="full"
              >
                <ScrollableContent>
                  <InvHeading
                    position="sticky"
                    top="0"
                    backgroundColor="base.750"
                    fontSize="large"
                    p="1"
                  >
                    {t('common.localSystem')}
                  </InvHeading>
                  {deps.map(({ name, version }, i) => (
                    <Flex
                      key={i}
                      flexDir="row"
                      w="full"
                      py="3"
                      px="1"
                      alignItems="center"
                      gap="12"
                      justifyContent="space-between"
                    >
                      <InvText>{name}</InvText>
                      <InvText>{version ? version : 'Not Installed'}</InvText>
                    </Flex>
                  ))}
                </ScrollableContent>
              </GridItem>
              <GridItem>
                <Flex flexDir="column" gap={3} mt="5rem" alignItems="center">
                  <Image src={InvokeLogoYellow} alt="invoke-logo" w="120px" />
                  {appVersion && <InvText>{`v${appVersion?.version}`}</InvText>}
                  <Grid templateColumns="repeat(2, 1fr)" gap="3">
                    <GridItem>
                      <Link fontSize="sm" href={githubLink} isExternal>
                        {t('common.githubLabel')}
                        <ExternalLinkIcon mx="2px" />
                      </Link>
                    </GridItem>
                    <GridItem>
                      <Link fontSize="sm" href={discordLink} isExternal>
                        {t('common.discordLabel')}
                        <ExternalLinkIcon mx="2px" />
                      </Link>
                    </GridItem>
                  </Grid>
                  <InvHeading fontSize="large">
                    {t('common.aboutHeading')}
                  </InvHeading>
                  <InvText fontSize="sm">{t('common.aboutDesc')}</InvText>
                  <Link isExternal href={websiteLink} fontSize="sm">
                    {websiteLink}
                  </Link>
                </Flex>
              </GridItem>
            </Grid>
          </InvModalBody>
          <InvModalFooter />
        </InvModalContent>
      </InvModal>
    </>
  );
};

export default memo(AboutModal);
