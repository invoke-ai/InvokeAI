import {
  AccordionButton,
  AccordionIcon,
  AccordionItem,
  AccordionPanel,
  Box,
  Flex,
} from '@chakra-ui/react';
import { ReactNode } from 'react';
import { Feature } from 'app/features';
import GuideIcon from 'common/components/GuideIcon';

export interface InvokeAccordionItemProps {
  header: string;
  content: ReactNode;
  feature?: Feature;
  additionalHeaderComponents?: ReactNode;
}

export default function InvokeAccordionItem(props: InvokeAccordionItemProps) {
  const { header, feature, content, additionalHeaderComponents } = props;

  return (
    <AccordionItem className="advanced-settings-item">
      <AccordionButton className="advanced-settings-header">
        <Flex width={'100%'} gap={'0.5rem'} align={'center'}>
          <Box flexGrow={1} textAlign={'left'}>
            {header}
          </Box>
          {additionalHeaderComponents}
          {feature && <GuideIcon feature={feature} />}
          <AccordionIcon />
        </Flex>
      </AccordionButton>
      <AccordionPanel className="advanced-settings-panel">
        {content}
      </AccordionPanel>
    </AccordionItem>
  );
}
