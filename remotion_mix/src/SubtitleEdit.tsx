import {
  OffthreadVideo,
  interpolate,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

type Word = {
  text: string;
  start: number;
  end: number;
};

type CaptionGroup = {
  start: number;
  end: number;
  words: Word[];
  breakBefore?: number;
};

const CAPTIONS: CaptionGroup[] = [
  {
    start: 0,
    end: 2.55,
    words: [
      { text: "So,", start: 0, end: 0.84 },
      { text: "if", start: 1.02, end: 1.3 },
      { text: "you", start: 1.3, end: 1.44 },
      { text: "like", start: 1.44, end: 1.66 },
      { text: "using", start: 1.66, end: 2.1 },
      { text: "AI", start: 2.1, end: 2.52 },
    ],
  },
  {
    start: 2.52,
    end: 3.98,
    words: [
      { text: "for", start: 2.52, end: 2.94 },
      { text: "whatever", start: 2.94, end: 3.18 },
      { text: "you", start: 3.18, end: 3.4 },
      { text: "need", start: 3.4, end: 3.62 },
      { text: "to,", start: 3.62, end: 3.9 },
    ],
  },
  {
    start: 4.05,
    end: 5.76,
    words: [
      { text: "or", start: 4.05, end: 4.1 },
      { text: "for", start: 4.1, end: 4.24 },
      { text: "work,", start: 4.24, end: 4.44 },
      { text: "school,", start: 4.44, end: 4.68 },
      { text: "whatever,", start: 5.0, end: 5.2 },
      { text: "but", start: 5.54, end: 5.72 },
    ],
  },
  {
    start: 5.7,
    end: 8.82,
    breakBefore: 6,
    words: [
      { text: "you", start: 5.72, end: 5.9 },
      { text: "don't", start: 5.9, end: 6.36 },
      { text: "like", start: 6.36, end: 6.52 },
      { text: "the", start: 6.52, end: 6.76 },
      { text: "fact", start: 6.76, end: 6.96 },
      { text: "that", start: 6.96, end: 7.18 },
      { text: "they're", start: 7.18, end: 7.44 },
      { text: "stealing", start: 7.44, end: 7.76 },
      { text: "all", start: 7.76, end: 8.06 },
      { text: "your", start: 8.06, end: 8.26 },
      { text: "data,", start: 8.26, end: 8.72 },
    ],
  },
  {
    start: 9.12,
    end: 10.38,
    words: [
      { text: "download", start: 9.2, end: 9.46 },
      { text: "CanopyChat,", start: 9.46, end: 10.3 },
    ],
  },
  {
    start: 10.4,
    end: 11.58,
    words: [
      { text: "it's", start: 10.46, end: 10.54 },
      { text: "completely", start: 10.54, end: 10.9 },
      { text: "private.", start: 10.9, end: 11.46 },
    ],
  },
];

const FONT = "Arial, Helvetica, sans-serif";

export const CanopyChatSubtitleEdit: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
  const time = frame / fps;

  const currentCaption = CAPTIONS.find(
    (caption) => time >= caption.start && time < caption.end,
  );

  const videoScale = interpolate(frame, [0, 510], [1.0, 1.035], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const videoX = Math.sin(frame / 58) * 2;

  return (
    <div
      style={{
        width,
        height,
        backgroundColor: "#0b0b11",
        overflow: "hidden",
        fontFamily: FONT,
      }}
    >
      <OffthreadVideo
        src={staticFile("subtitle-base.mp4")}
        volume={1}
        style={{
          width: "100%",
          height: "100%",
          objectFit: "cover",
          transform: `translateX(${videoX}px) scale(${videoScale})`,
        }}
      />

      <div
        style={{
          position: "absolute",
          inset: 0,
          background:
            "linear-gradient(180deg, rgba(0,0,0,0.05) 48%, rgba(0,0,0,0.08) 62%, rgba(0,0,0,0.72) 100%)",
          pointerEvents: "none",
        }}
      />

      <div
        style={{
          position: "absolute",
          top: 42,
          left: 28,
          padding: "7px 11px 6px",
          border: "1px solid rgba(255,255,255,0.6)",
          borderRadius: 999,
          color: "white",
          fontSize: 15,
          fontWeight: 800,
          letterSpacing: 1.2,
          background: "rgba(0,0,0,0.24)",
        }}
      >
        CANOPYCHAT
      </div>

      {currentCaption ? (
        <div
          style={{
            position: "absolute",
            left: 28,
            right: 28,
            bottom: 148,
            color: "white",
            fontSize: 31,
            lineHeight: 1.08,
            fontWeight: 850,
            letterSpacing: -0.7,
            textShadow: "0 3px 12px rgba(0,0,0,0.9)",
            transform: `scale(${interpolate(
              frame,
              [currentCaption.start * fps, currentCaption.start * fps + 8],
              [0.92, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
            )})`,
            transformOrigin: "left bottom",
          }}
        >
          <div
            style={{
              display: "inline",
              padding: "7px 12px 9px",
              borderRadius: 14,
              background: "rgba(5,5,10,0.62)",
              boxDecorationBreak: "clone",
              WebkitBoxDecorationBreak: "clone",
            } as React.CSSProperties}
          >
            {currentCaption.words.map((word, index) => {
              const active = time >= word.start && time < word.end;
              const shouldBreak = currentCaption.breakBefore === index;
              return (
                <span key={`${word.text}-${index}`}>
                  {shouldBreak ? <br /> : index > 0 ? " " : null}
                  <span
                    style={{
                      color: active ? "#ff4ca3" : "#ffffff",
                      WebkitTextStroke: active ? "0.25px #ff4ca3" : undefined,
                      display: "inline-block",
                      transform: active ? "scale(1.04)" : "scale(1)",
                      transition: "transform 80ms linear",
                    }}
                  >
                    {word.text}
                  </span>
                </span>
              );
            })}
          </div>
        </div>
      ) : null}

      {time >= 11.58 && time < 13.75 ? (
        <div
          style={{
            position: "absolute",
            left: 28,
            right: 28,
            bottom: 158,
            color: "#ff4ca3",
            fontSize: 38,
            lineHeight: 1,
            fontWeight: 900,
            letterSpacing: -1.4,
            textAlign: "center",
            textShadow: "0 4px 18px rgba(0,0,0,0.9)",
            opacity: interpolate(time, [11.58, 11.8, 13.4, 13.75], [0, 1, 1, 0], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
            transform: `scale(${interpolate(time, [11.58, 11.82], [0.82, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            })})`,
          }}
        >
          COMPLETELY PRIVATE.
        </div>
      ) : null}
    </div>
  );
};
