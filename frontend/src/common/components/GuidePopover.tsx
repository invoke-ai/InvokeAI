import {
    Popover,
    PopoverArrow,
    PopoverContent,
    PopoverTrigger,
    Flex
  } from "@chakra-ui/react";
import {
    SystemState
} from "../../features/system/systemSlice";
import { useAppSelector } from '../../app/store';
import { RootState } from '../../app/store';
import { createSelector } from '@reduxjs/toolkit';
import { ReactElement } from "react";
import { Guides } from "../../app/guides";

type GuideProps = {
    children: ReactElement;
    feature: keyof typeof Guides;
  };


  const systemSelector = createSelector(
    (state: RootState) => state.system,
    (system: SystemState) => system.shouldDisplayGuides
  );
 {/* const shouldShowGuideVisuals = false*/} 

const GuidePopover = ({ children, feature }: GuideProps) => {
    const shouldDisplayGuides = useAppSelector(systemSelector);
    const { text } = Guides[feature];
    if (shouldDisplayGuides) {   
        return (
            <Popover trigger={"hover"} placement='right'>
                <PopoverTrigger>{children}</PopoverTrigger>
                <PopoverContent width={"auto"}>
                    <PopoverArrow />
                    {/*  if (shouldShowGuideVisuals) {   
                        <PopoverHeader>
                            <Flex alignItems={"center"} gap={2}>
                                <Image src={guideImage} alt={text} />
                            </Flex>
                        </PopoverHeader>}*/}
                
                    
                        <Flex alignItems={"center"} gap={2} p={4}>
                            {text} 
                        </Flex>
                        {/* Commenting out learn more button and link, until documentation links are ready.*/} 
                        {/* 
                        <Flex alignItems={"right"} gap={2} p={4}>
                            <Spacer />
                            <Spacer />
                            <Spacer />
                            <Spacer />
                            <Spacer />
                            <Spacer />   
                            <Link href={href} isExternal>
                                <Button leftIcon={<ExternalLinkIcon />} colorScheme='teal' variant='solid'>
                                Learn More
                                </Button>
                            </Link> 
                        </Flex>     
                        */}      
                </PopoverContent>
            </Popover>
            );
        }
        return
    }
    
export default GuidePopover;