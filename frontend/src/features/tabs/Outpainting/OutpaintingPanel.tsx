import { Box } from "@chakra-ui/react";
import React from "react";
import { Feature } from "../../../app/features";
import { RootState, useAppSelector } from "../../../app/store";
import SeedOptions from "../../options/AdvancedOptions/Seed/SeedOptions";
import CanvasOptions from "../../options/CanvasOptions";
import MainAdvancedOptions from "../../options/MainOptions/MainAdvancedOptions";
import MainOptions from "../../options/MainOptions/MainOptions";
import OptionsAccordion from "../../options/OptionsAccordion";
import ProcessButtons from "../../options/ProcessButtons/ProcessButtons";
import PromptInput from "../../options/PromptInput/PromptInput";

export default function OutpaintingPanel() {
  const showAdvancedOptions = useAppSelector(
    (state: RootState) => state.options.showAdvancedOptions  
  );

  const outpaintingAccordions = {
    seed: {
      header: (
        <Box flex="1" textAlign="left">
          Seed   
        </Box>
      ),
      feature: Feature.SEED,
      options: <SeedOptions />,
    },
    canvas_size: {
      header: (
        <Box flex="1" textAlign="left">
          Canvas Size
        </Box>
      ),
      feature: Feature.OTHER,
      options: <CanvasOptions />,
    }
  }

  return (
    <div className="outpainting-panel">
      <PromptInput />
      <ProcessButtons />
      <MainOptions />
      <MainAdvancedOptions />
      {showAdvancedOptions && (
        <OptionsAccordion accordionInfo={outpaintingAccordions} />
      )}
    </div>
  )
}