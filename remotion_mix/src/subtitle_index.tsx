import { Composition } from "remotion";
import { CanopyChatSubtitleEdit } from "./SubtitleEdit";

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="CanopyChatSubtitleEdit"
      component={CanopyChatSubtitleEdit}
      durationInFrames={516}
      fps={30}
      width={576}
      height={1024}
    />
  );
};
