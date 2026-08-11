const GLYPHS = {
  "0": ["111", "101", "101", "101", "101", "101", "111"],
  "1": ["010", "110", "010", "010", "010", "010", "111"],
  "7": ["111", "001", "001", "010", "010", "100", "100"],
  "8": ["111", "101", "101", "111", "101", "101", "111"],
  "9": ["111", "101", "101", "111", "001", "001", "111"],
  A: ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
  D: ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
  J: ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
  K: ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
  Q: ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
  R: ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
  V: ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
};

export function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

export function bitmapGlyph(text, x, y, cell, color) {
  let cursor = x;
  const rectangles = [];
  for (const character of text) {
    const rows = GLYPHS[character];
    if (!rows) throw new Error(`Missing vector glyph: ${character}`);
    rows.forEach((row, rowIndex) => {
      [...row].forEach((bit, columnIndex) => {
        if (bit === "1") {
          rectangles.push(
            `<rect x="${cursor + columnIndex * cell}" y="${y + rowIndex * cell}" width="${cell}" height="${cell}"/>`,
          );
        }
      });
    });
    cursor += (rows[0].length + 1) * cell;
  }
  return `<g fill="${color}">${rectangles.join("")}</g>`;
}

export function suitMark(suit, x, y, size, color) {
  const centerX = x + size / 2;
  const centerY = y + size / 2;
  if (suit === "diamonds") {
    return `<path fill="${color}" d="M ${centerX} ${y} L ${x + size} ${centerY} L ${centerX} ${y + size} L ${x} ${centerY} Z"/>`;
  }
  if (suit === "hearts") {
    return `<path fill="${color}" d="M ${centerX} ${y + size} C ${x + size * 0.1} ${y + size * 0.66}, ${x - size * 0.04} ${y + size * 0.25}, ${x + size * 0.24} ${y + size * 0.13} C ${x + size * 0.42} ${y + size * 0.05}, ${centerX} ${y + size * 0.2}, ${centerX} ${y + size * 0.3} C ${centerX} ${y + size * 0.2}, ${x + size * 0.58} ${y + size * 0.05}, ${x + size * 0.76} ${y + size * 0.13} C ${x + size * 1.04} ${y + size * 0.25}, ${x + size * 0.9} ${y + size * 0.66}, ${centerX} ${y + size} Z"/>`;
  }
  if (suit === "clubs") {
    return `<g fill="${color}"><circle cx="${centerX}" cy="${y + size * 0.32}" r="${size * 0.24}"/><circle cx="${x + size * 0.3}" cy="${y + size * 0.57}" r="${size * 0.24}"/><circle cx="${x + size * 0.7}" cy="${y + size * 0.57}" r="${size * 0.24}"/><path d="M ${centerX - size * 0.09} ${y + size * 0.55} L ${centerX + size * 0.09} ${y + size * 0.55} L ${x + size * 0.65} ${y + size} L ${x + size * 0.35} ${y + size} Z"/></g>`;
  }
  return `<path fill="${color}" d="M ${centerX} ${y} C ${x + size * 0.62} ${y + size * 0.26}, ${x + size} ${y + size * 0.42}, ${x + size} ${y + size * 0.65} C ${x + size} ${y + size * 0.88}, ${x + size * 0.72} ${y + size * 0.9}, ${centerX + size * 0.07} ${y + size * 0.72} L ${x + size * 0.68} ${y + size} L ${x + size * 0.32} ${y + size} L ${centerX - size * 0.07} ${y + size * 0.72} C ${x + size * 0.28} ${y + size * 0.9}, ${x} ${y + size * 0.88}, ${x} ${y + size * 0.65} C ${x} ${y + size * 0.42}, ${x + size * 0.38} ${y + size * 0.26}, ${centerX} ${y} Z"/>`;
}

function oneFrenchCorner(width, height, glyph, suit, variant) {
  const cornerWidth = width * 0.27;
  const cornerHeight = height * 0.35;
  const color = suit === "diamonds" || suit === "hearts" ? "#d71935" : "#111318";
  const cell = width * (variant === "holdout" ? 0.0205 : 0.0185);
  const glyphWidth = [...glyph].reduce(
    (sum, character) => sum + (GLYPHS[character][0].length + 1) * cell,
    -cell,
  );
  const glyphX = (cornerWidth - glyphWidth) / 2;
  const glyphY = height * (variant === "holdout" ? 0.035 : 0.027);
  const markSize = width * (variant === "holdout" ? 0.135 : 0.12);
  const border =
    variant === "holdout"
      ? `<path d="M 1 ${cornerHeight - 1} L 1 1 L ${cornerWidth - 1} 1" fill="none" stroke="#d8d5cb" stroke-width="1"/>`
      : "";
  return `<g>
    <rect x="0" y="0" width="${cornerWidth}" height="${cornerHeight}" fill="#fffef9"/>
    ${border}
    ${bitmapGlyph(glyph, glyphX, glyphY, cell, color)}
    ${suitMark(suit, (cornerWidth - markSize) / 2, height * 0.185, markSize, color)}
  </g>`;
}

export function frenchCornerOverlay(width, height, glyph, suit, variant) {
  const corner = oneFrenchCorner(width, height, glyph, suit, variant);
  return `${corner}<g transform="translate(${width} ${height}) rotate(180)">${corner}</g>`;
}
