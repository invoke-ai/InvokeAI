import {
    Popover,
    PopoverArrow,
    PopoverContent,
    PopoverTrigger,
    PopoverHeader,
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

const GuidePopover = ({ children, feature }: GuideProps) => {
    const shouldDisplayGuides = useAppSelector(systemSelector);
    const { text } = Guides[feature];
    if (shouldDisplayGuides) {   
        return (
            <Popover trigger={"hover"} placement='right'>
                <PopoverTrigger>{children}</PopoverTrigger>
                    <PopoverHeader>
                        <PopoverContent width={"auto"}>
                            <PopoverArrow />
                                <Flex alignItems={"center"} gap={2} p={4}>
                                    {text} 
                                </Flex>
                        </PopoverContent>
                    </PopoverHeader>
            </Popover>
            );
        }
        return (
        /**
 * This is how I was able to get the Guides option to not hide UI elements (ensuring that the 'child' is displayed)
 * However, it seems odd to have a non-functional "popover" wrapping all of the UI elements when that's off, so this probably needs a #todo to fix
 */

            <Popover>
                 <PopoverTrigger>{children}</PopoverTrigger>
            </Popover>)
    }
    
export default GuidePopover;