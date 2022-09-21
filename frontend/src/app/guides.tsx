/**
 * Each feature with a UI element is listed here.
 * 
 * 'text' is the short help text for that feature.
 * 'href' is the link to the docs page for that feature.
 * 'visual' is used if an image or animated asset has been referenced for display in the tooltip.
 */

 export const Guides = {
    Prompt: {
      text: "This field will take all prompt text, including both content and stylistic terms. CLI Commands will not work in the prompt.",
      href: "link/to/docs/feature3.html",
      guideImage: "asset/path.gif"
    },
    Gallery: {
      text: "As new invocations are generated, files from the output directory will be displayed here. Generations have additional options to configure new generations.",
      href: "link/to/docs/feature3.html",
      guideImage: "asset/path.gif"
    },
    Steps: {
      text: "This field controls the number of denoising steps used by InvokeAI. This can result in additional detail, depending on the sampler.",
      href: "link/to/docs/feature3.html",
      guideImage: "asset/path.gif"
    },
    Resolution: {
      text: "The Height and Width of generations can be controlled here. If you experience errors, you may be generating an image too large for your system.",
      href: "link/to/docs/feature3.html",
      guideImage: "asset/path.gif"
    },
    ConfigScale: {
      text: "The Config Scale value dictates how constrained the invocation is by the prompt. Higher values may more closely match the prompt, at the expense of flexibility in generating a more aesthetic output.",
      href: "link/to/docs/feature3.html",
      guideImage: "asset/path.gif"
    },
    Seed: {
      text: "Seed values provide an initial set of noise which guide the denoising process. The same seed & configuration settings should reproduce the same output.",
      href: "link/to/docs/feature3.html",
      guideImage: "asset/path.gif"
    },
    Seamless: {
      text: "A neat feature that will more often result in repeating patterns in outputs.",
      href: "link/to/docs/feature3.html",
      guideImage: "asset/path.gif"
    },
    ESRGAN: {
      text: "The ESRGAN setting can be used to increase the output resolution without requiring a higher width/height in the initial generation.",
      href: "link/to/docs/feature1.html",
      guideImage: "asset/path.gif"
    },
    FaceCorrection: {
      text: "Using ESRGAN or CodeFormer, Face Correction will attempt to identify faces in outputs, and correct any defects/abnormalities. Higher values will apply a stronger corrective pressure on outputs.",
      href: "link/to/docs/feature2.html",
      guideImage: "asset/path.gif"
    },
    ImageToImage: {
      text: "ImageToImage allows the upload of an initial image, which InvokeAI will use to guide the generation process, along with a prompt. A lower value for this setting will more closely resemble the original image. Values between 0-1 are accepted, and a range of .25-.75 is recommended ",
      href: "link/to/docs/feature3.html",
      guideImage: "asset/path.gif"
    },
    Sampler: {
      text: "This setting allows for different denoising samplers to be used, which can alter the generated image.",
      href: "link/to/docs/feature3.html",
      guideImage: "asset/path.gif"
    },
    Variations: {
      text: "Using an initial seed, variations will alter the noise slightly to provide interesting variations on the output image. Higher values more significantly vary the noise, and a recommended value for simple variations is 0.2",
      href: "link/to/docs/feature3.html",
      guideImage: "asset/path.gif"
    },
    GeneralHelp: {
      text: "For more help, read the full docs here.",
      href: "link/to/docs/feature3.html",
      guideImage: "asset/path.gif"
    },

  };
  