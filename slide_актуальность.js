const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";

const BG       = "0D1B2A";
const CARD_BG  = "132233";
const ACCENT   = "00B4D8";
const WHITE    = "FFFFFF";
const MUTED    = "7EA8C4";

const slide = pres.addSlide();
slide.background = { color: BG };

// ── Заголовок ─────────────────────────────────────────────────────────────────
slide.addText("Актуальность", {
  x: 0.5, y: 0.22, w: 9, h: 0.72,
  fontSize: 38, fontFace: "Georgia", bold: true,
  color: WHITE, align: "left", margin: 0,
});

// тонкая акцентная линия под заголовком
slide.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 0.97, w: 1.6, h: 0.045,
  fill: { color: ACCENT }, line: { color: ACCENT },
});

// подзаголовок
slide.addText("Геолокализация изображений в задачах безопасности", {
  x: 0.5, y: 1.08, w: 9, h: 0.38,
  fontSize: 13, fontFace: "Calibri", color: MUTED, align: "left", margin: 0,
});

// ── Карточки ──────────────────────────────────────────────────────────────────
const cards = [
  {
    num: "01",
    title: "ОРД: точка съёмки",
    body: "Определение местоположения точки съёмки фото и видео при проведении оперативно-розыскных мероприятий",
  },
  {
    num: "02",
    title: "Скомпрометированные камеры",
    body: "Определение географических координат сетевых камер видеонаблюдения, захваченных злоумышленниками",
  },
  {
    num: "03",
    title: "Отслеживание передвижений",
    body: "Мониторинг перемещения техники и личного состава по фотоматериалам из открытых источников",
  },
];

const cardW = 2.85;
const cardH = 3.35;
const cardY = 1.65;
const gap   = 0.23;
const startX = (10 - (cardW * 3 + gap * 2)) / 2;

cards.forEach((card, i) => {
  const x = startX + i * (cardW + gap);

  // фон карточки
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y: cardY, w: cardW, h: cardH,
    fill: { color: CARD_BG },
    line: { color: "1E3A52", width: 0.8 },
    shadow: { type: "outer", color: "000000", blur: 12, offset: 4, angle: 135, opacity: 0.35 },
  });

  // акцентная полоса сверху карточки
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y: cardY, w: cardW, h: 0.065,
    fill: { color: ACCENT }, line: { color: ACCENT },
  });

  // номер
  slide.addText(card.num, {
    x: x + 0.22, y: cardY + 0.2, w: 0.75, h: 0.6,
    fontSize: 28, fontFace: "Georgia", bold: true,
    color: ACCENT, align: "left", margin: 0,
  });

  // заголовок карточки
  slide.addText(card.title, {
    x: x + 0.18, y: cardY + 0.88, w: cardW - 0.36, h: 0.72,
    fontSize: 14, fontFace: "Calibri", bold: true,
    color: WHITE, align: "left", margin: 0,
  });

  // горизонтальный разделитель
  slide.addShape(pres.shapes.RECTANGLE, {
    x: x + 0.18, y: cardY + 1.65, w: cardW - 0.36, h: 0.025,
    fill: { color: "1E3A52" }, line: { color: "1E3A52" },
  });

  // текст описания
  slide.addText(card.body, {
    x: x + 0.18, y: cardY + 1.78, w: cardW - 0.36, h: 1.45,
    fontSize: 12, fontFace: "Calibri",
    color: MUTED, align: "left", valign: "top", margin: 0,
  });
});

pres.writeFile({ fileName: "актуальность.pptx" }).then(() => {
  console.log("Saved: актуальность.pptx");
});
