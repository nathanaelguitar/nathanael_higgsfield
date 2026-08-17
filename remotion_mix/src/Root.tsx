import { Audio, Composition, Img, OffthreadVideo, Sequence } from "remotion";
import { staticFile } from "remotion";

const FPS = 24;
const SOURCE_DURATION_IN_FRAMES = 362;
const MUSIC_HOLD_IN_FRAMES = 1 * FPS;

// End frames for the spoken blocks in the source trailer. Each block is
// followed by a two-second hold using the opening of the instrumental.
const LINE_END_FRAMES = [36, 94, 142, 182, 220, 266, 295, 319, 362] as const;
const DURATION_IN_FRAMES =
  SOURCE_DURATION_IN_FRAMES + LINE_END_FRAMES.length * MUSIC_HOLD_IN_FRAMES;

const videoStyle = {
  width: "100%",
  height: "100%",
  objectFit: "cover" as const,
};

const CanopyChatTrailer: React.FC = () => {
  const timeline: React.ReactNode[] = [];
  let outputFrame = 0;
  let sourceStartFrame = 0;

  LINE_END_FRAMES.forEach((sourceEndFrame, index) => {
    const spokenDuration = sourceEndFrame - sourceStartFrame;

    timeline.push(
      <Sequence
        key={`spoken-${index}`}
        from={outputFrame}
        durationInFrames={spokenDuration}
      >
        <OffthreadVideo
          src={staticFile("base-trailer.mp4")}
          startFrom={sourceStartFrame}
          endAt={sourceEndFrame}
          volume={1}
          style={videoStyle}
        />
      </Sequence>,
    );
    outputFrame += spokenDuration;

    timeline.push(
      <Sequence
        key={`music-hold-${index}`}
        from={outputFrame}
        durationInFrames={MUSIC_HOLD_IN_FRAMES}
      >
        <Img
          src={staticFile(`freeze-${index + 1}.png`)}
          style={videoStyle}
        />
        <Audio
          src={staticFile("music_instrumental.wav")}
          startFrom={0}
          endAt={MUSIC_HOLD_IN_FRAMES}
          volume={0.32}
        />
      </Sequence>,
    );
    outputFrame += MUSIC_HOLD_IN_FRAMES;
    sourceStartFrame = sourceEndFrame;
  });

  return <>{timeline}</>;
};

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="CanopyChatTrailer"
      component={CanopyChatTrailer}
      durationInFrames={DURATION_IN_FRAMES}
      fps={FPS}
      width={720}
      height={1280}
    />
  );
};
