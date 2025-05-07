// let animationInterval;

// function animateTraffic(filteredGeojson) {
//   // Limpiar animaciones previas
//   clearInterval(animationInterval);
//   d3.select(map.getCanvasContainer()).selectAll('circle.traffic-dot').remove();

//   // Crear SVG si no existe
//   let svg = d3.select(map.getContainer()).select('svg');
//   if (svg.empty()) {
//     svg = d3.select(map.getContainer())
//       .append('svg')
//       .style('position', 'absolute')
//       .style('top', 0)
//       .style('left', 0)
//       .style('width', '100%')
//       .style('height', '100%')
//       .style('pointer-events', 'none')
//       .style('z-index', 2);
//   }

//   // Escala de colores: azul (bajo) → rojo (alto)
//   const colorScale = d3.scaleSequential(d3.interpolateTurbo).domain([0, 1]);

//   // Animación por ciclo
//   function renderCycle() {
//     svg.selectAll('circle.traffic-dot').remove();
//     let totalDots = 0;

//     filteredGeojson.features.forEach(f => {
//       const coords = f.geometry.coordinates;
//       const intensity = +f.properties.activity_index;

//       // Validación
//       if (!Array.isArray(coords) || coords.length < 2 || isNaN(intensity)) return;

//       const steps = Math.floor(intensity * 100);  // puedes ajustar el factor aquí

//       for (let i = 0; i < steps; i++) {
//         const idx = Math.floor(Math.random() * (coords.length - 1));
//         if (!Array.isArray(coords[idx]) || coords[idx].length !== 2) continue;

//         const start = map.project(coords[idx]);
//         const end = map.project(coords[idx + 1]);

//         if (!start || !end) return;

//         totalDots++;

//         svg.append('circle')
//           .attr('class', 'traffic-dot')
//           .attr('r', 2)
//           .attr('cx', start.x)
//           .attr('cy', start.y)
//           .style('fill', colorScale(intensity))
//           .style('opacity', 0.8)
//           .transition()
//           .duration(3000 - Math.min(intensity, 1) * 2000)
//           .ease(d3.easeLinear)
//           .attr('cx', end.x)
//           .attr('cy', end.y)
//           .remove();
//       }
//     });

//     console.log('Puntos animados este ciclo:', totalDots);
//   }

//   renderCycle();
//   animationInterval = setInterval(renderCycle, 3000);
// }


// static/js/traffic_animator.js

let animationTime = 0;
let animationInterval;
let flatAnimatedPoints = [];

// Actualiza los puntos animados
function updateAnimatedTrafficPoints(filteredFeatures) {
  flatAnimatedPoints = [];

  filteredFeatures.forEach(feature => {
    const geometry = feature.geometry;
    const intensity = feature.properties.activity_index;
    const steps = Math.floor(intensity * 100);

    const coords = geometry.coordinates;
    if (coords.length < 2) return;

    for (let i = 0; i < steps; i++) {
      const t = (i / steps + animationTime) % 1;
      const index = t * (coords.length - 1);
      const lower = Math.floor(index);
      const upper = Math.ceil(index);
      const frac = index - lower;

      const p0 = coords[lower];
      const p1 = coords[upper];
      if (!p0 || !p1) continue;

      const x = p0[0] + (p1[0] - p0[0]) * frac;
      const y = p0[1] + (p1[1] - p0[1]) * frac;

      flatAnimatedPoints.push({
        position: [x, y],
        intensity
      });
    }
  });
}

// Capa de animación
function getTrafficAnimationLayer() {
  return new deck.ScatterplotLayer({
    id: 'traffic-animation',
    data: flatAnimatedPoints,
    getPosition: d => d.position,
    getFillColor: d => {
      const color = d3.color(d3.interpolateTurbo(d.intensity)).rgb();
      return [color.r, color.g, color.b];
    },
    getRadius: d => 10,
    radiusMinPixels: 1,
    radiusMaxPixels: 8,
    opacity: 0.6,
    pickable: false,
    updateTriggers: {
      data: flatAnimatedPoints
    }
  });
}

// Función principal
window.startTrafficAnimation = function (deckOverlay, getFilteredFeatures) {
  if (animationInterval) clearInterval(animationInterval);

  animationInterval = setInterval(() => {
    animationTime = (animationTime + 0.01) % 1;
    const filtered = getFilteredFeatures();
    updateAnimatedTrafficPoints(filtered);

    const existingLayers = (deckOverlay.props?.layers || []).filter(l => l.id !== 'traffic-animation');
    const newLayer = getTrafficAnimationLayer();

    deckOverlay.setProps({
      layers: [...existingLayers, newLayer]
    });
  }, 50);
};

window.stopTrafficAnimation = function () {
  clearInterval(animationInterval);
};

