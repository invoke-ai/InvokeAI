import {
  AccordionButton,
  AccordionIcon,
  AccordionItem,
  AccordionPanel,
  Box,
  Flex,
} from '@chakra-ui/react';
import { Feature } from 'app/features';
import GuideIcon from 'common/components/GuideIcon';
import { ReactNode } from 'react';

export interface InvokeAccordionItemProps {
  header: string;
  content: ReactNode;
  feature?: Feature;
  additionalHeaderComponents?: ReactNode;
}

export default function InvokeAccordionItem(props: InvokeAccordionItemProps) {
  const { header, feature, content, additionalHeaderComponents } = props;

  return (
    <AccordionItem>
      <AccordionButton>
        <Flex width="100%" gap={2} align="center">
          <Box flexGrow={1} textAlign="start">
            {header}
          </Box>
          {additionalHeaderComponents}
          {feature && <GuideIcon feature={feature} />}
          <AccordionIcon />
        </Flex>
      </AccordionButton>
      <AccordionPanel>{content}</AccordionPanel>
    </AccordionItem>
  );
}
