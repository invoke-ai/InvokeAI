import { ExternalLinkIcon } from '@chakra-ui/icons';
import {
  Flex,
  Grid,
  GridItem,
  Image,
  Link,
  Table,
  TableContainer,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
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
import { fromPairs, map, toPairs } from 'lodash-es';
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
  const { data: appDeps } = useGetAppDepsQuery();
  const { data: appVersion } = useGetAppVersionQuery();
  const appDeps_arr = map(toPairs(appDeps), (d) => fromPairs([d]));

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
            <ScrollableContent>
              <Grid templateColumns="repeat(2, 1fr)">
                <GridItem backgroundColor="base.750" borderRadius="base">
                  <TableContainer>
                    <Table variant="unstyled" w="50%">
                      <Thead>
                        <Tr>
                          <Th fontSize="medium" color="white">
                            {t('common.localSystem')}
                          </Th>
                        </Tr>
                      </Thead>
                      {appDeps_arr.map((deps, index) => (
                        <Tbody key={index}>
                          {Object.entries(deps).map(([key, value]) => (
                            <Tr key={key} fontSize="sm" color="white">
                              <Td py={2}>{key}</Td>
                              <Td py={2}>{value ? value : 'Not Installed'}</Td>
                            </Tr>
                          ))}
                        </Tbody>
                      ))}
                    </Table>
                  </TableContainer>
                </GridItem>
                <GridItem>
                  <Flex flexDir="column" gap={3} mt="5rem" alignItems="center">
                    <Image src={InvokeLogoYellow} alt="invoke-logo" w="120px" />
                    {appVersion && (
                      <InvText>{`v${appVersion?.version}`}</InvText>
                    )}
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
                    <Link href={websiteLink} fontSize="sm">
                      {websiteLink}
                    </Link>
                  </Flex>
                </GridItem>
              </Grid>
            </ScrollableContent>
          </InvModalBody>
          <InvModalFooter />
        </InvModalContent>
      </InvModal>
    </>
  );
};

export default memo(AboutModal);
