import {
  AccordionButton,
  AccordionIcon,
  AccordionItem,
  AccordionPanel,
  Box,
  Flex,
} from '@chakra-ui/react';
import GuideIcon from 'common/components/GuideIcon';
import { ParametersAccordionItem } from '../ParametersAccordion';

type InvokeAccordionItemProps = {
  accordionItem: ParametersAccordionItem;
};

export default function InvokeAccordionItem({
  accordionItem,
}: InvokeAccordionItemProps) {
  const { header, feature, content, additionalHeaderComponents } =
    accordionItem;

  return (
    <AccordionItem>
      <AccordionButton>
        <Flex width="100%" gap={2} align="center">
          <Box flexGrow={1} textAlign="start">
            {header}
          </Box>
          {additionalHeaderComponents}
          {/* {feature && <GuideIcon feature={feature} />} */}
          <AccordionIcon />
        </Flex>
      </AccordionButton>
      <AccordionPanel p={4}>{content}</AccordionPanel>
    </AccordionItem>
  );
}
