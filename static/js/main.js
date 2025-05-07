// mapboxgl.accessToken = 'pk.eyJ1Ijoib21lcnV4MzIiLCJhIjoiY205cmI2ampiMWNjMzJscHo4MDNibzMwaCJ9.J9ioizrRkVYVwZCWVoDFBw';

// const map = new mapboxgl.Map({
//   container: 'map',
//   style: 'mapbox://styles/mapbox/dark-v11',
//   center: [-122.33, 47.61],
//   zoom: 12
// });

// const deckOverlay = new deck.MapboxOverlay({
//   layers: []
// });

// map.on('load', () => {
//   map.addControl(deckOverlay);
//   fetchAndRenderTraffic();
// });

// function fetchAndRenderTraffic() {
//   fetch('/api/roads?hour_start=0&hour_end=23')
//     .then(res => res.json())
//     .then(data => {
//       const lineLayer = new deck.LineLayer({
//         id: 'traffic-lines',
//         data: data.features,
//         getSourcePosition: d => d.geometry.coordinates[0],
//         getTargetPosition: d => d.geometry.coordinates[d.geometry.coordinates.length - 1],
//         getColor: d => {
//           const a = d.properties.activity_index || 0;
//           return [255, 140 - a * 800, 0]; // rojo-naranja según actividad
//         },
//         getWidth: d => (d.properties.activity_index || 0.01) * 100,
//         pickable: true,
//         autoHighlight: true
//       });

//       deckOverlay.setProps({
//         layers: [lineLayer]
//       });
//     });
// }


// const layerId = 'traffic-layer';
// let mapData = {
//   type: "FeatureCollection",
//   features: []
// };

// const hourLabels = {
//   1: "00:00 – 06:00",
//   2: "06:00 – 10:00",
//   3: "10:00 – 15:00",
//   4: "15:00 – 19:00",
//   5: "19:00 – 00:00"
// };

// document.getElementById('hour-range').addEventListener('input', function () {
//   const selectedHour = this.value;
//   document.getElementById('selected-hour').textContent = hourLabels[selectedHour];
//   fetchAndRenderData();  // o cualquier función que actualice tu mapa
// });

// document.getElementById('activity-range').addEventListener('input', function () {
//   document.getElementById('selected-activity').textContent = this.value;
//   updateVisualization();
// });

// document.getElementById('day-selector').addEventListener('change', updateVisualization);
// document.getElementById('typeFilter').addEventListener('change', updateVisualization);

// // Visualización dinámica combinando todos los filtros
// function updateVisualization() {
//   const hour = parseInt(document.getElementById('hour-range').value);
//   const selectedDays = Array.from(document.getElementById('day-selector').selectedOptions).map(o => parseInt(o.value));
//   const selectedRoadType = document.getElementById('typeFilter').value;
//   const activityThreshold = parseFloat(document.getElementById('activity-range').value);

//   const filteredFeatures = mapData.features.filter(feature => {
//     const props = feature.properties;
//     return (
//       selectedDays.includes(props.day_code) &&
//       props.hour_code === hour &&
//       props.road_type === selectedRoadType &&
//       props.activity_index >= activityThreshold
//     );
//   });

//   const filteredGeoJSON = {
//     type: "FeatureCollection",
//     features: filteredFeatures
//   };

//   if (map.getLayer(layerId)) {
//     map.removeLayer(layerId);
//     map.removeSource(layerId);
//   }

//   map.addSource(layerId, {
//     type: 'geojson',
//     data: filteredGeoJSON
//   });

//   map.addLayer({
//     id: layerId,
//     type: 'line',
//     source: layerId,
//     paint: {
//       'line-width': 2,
//       'line-color': [
//         'interpolate',
//         ['linear'],
//         ['get', 'activity_index'],
//         0, '#2DC4B2',
//         0.02, '#3BB3C3',
//         0.04, '#669EC4',
//         0.06, '#8B88B6',
//         0.08, '#A2719B',
//         0.1, '#AA5E79'
//       ]
//     }
//   });

//   animateTraffic(filteredGeoJSON);
//   drawChart(filteredGeoJSON);
// }

// // Obtener datos desde la API Flask y almacenarlos en `mapData`
// function fetchAndRenderData() {
//   const url = new URL("/api/roads", window.location.origin);
//   url.searchParams.append("hour_start", 0);
//   url.searchParams.append("hour_end", 23);

//   fetch(url)
//     .then(res => res.json())
//     .then(data => {
//       mapData = data;
//       updateVisualization();
//     });
// }

// map.on("load", () => {
//   const container = map.getCanvasContainer();
//   const svg = d3.select(container).select("svg");
//   if (svg.empty()) {
//     d3.select(container).append("svg");
//   }
//   fetchAndRenderData();
// });


// function loadDailyPoints() {
//   fetch('/api/daily_activity')
//     .then(res => res.json())
//     .then(data => {
//       if (map.getLayer('activity-circles')) {
//         map.removeLayer('activity-circles');
//         map.removeSource('activity-circles');
//       }

//       map.addSource('activity-circles', {
//         type: 'geojson',
//         data: data
//       });

//       map.addLayer({
//         id: 'activity-circles',
//         type: 'circle',
//         source: 'activity-circles',
//         paint: {
//           'circle-radius': [
//             'interpolate',
//             ['linear'],
//             ['get', 'activity_index'],
//             0, 2,
//             0.1, 10,
//             1, 30
//           ],
//           'circle-color': '#FF5733',
//           'circle-opacity': 0.6
//         }
//       });
//     });
// }

// // Llama esto cuando cargue el mapa
// map.on('load', () => {
//   loadDailyPoints();
// });




// static/js/main.js

mapboxgl.accessToken = 'pk.eyJ1Ijoib21lcnV4MzIiLCJhIjoiY205cmI2ampiMWNjMzJscHo4MDNibzMwaCJ9.J9ioizrRkVYVwZCWVoDFBw';

// Aquí cargamos desde el CDN
const MapboxOverlay = deck.MapboxOverlay;

const map = new mapboxgl.Map({
  container: 'map',
  style: 'mapbox://styles/mapbox/dark-v11',
  center: [-122.33, 47.61],
  zoom: 12
});

const deckOverlay = new MapboxOverlay({ layers: [] });

map.on('load', () => {
  map.addControl(deckOverlay);
  fetchAndRenderData();
  loadDailyPoints(); // Mueve esto aquí para que se ejecute una vez
});

function fetchAndRenderData() {
  fetch('/api/roads?hour_start=0&hour_end=23')
    .then(res => res.json())
    .then(data => {
      const filteredFeatures = filterTrafficData(data.features);
      startTrafficAnimation(deckOverlay, () => filteredFeatures);
    });
}

function filterTrafficData(features) {
  const hour = parseInt(document.getElementById('hour-range').value);
  const selectedDays = Array.from(document.getElementById('day-selector').selectedOptions).map(o => parseInt(o.value));
  const selectedRoadType = document.getElementById('typeFilter').value;
  const activityThreshold = parseFloat(document.getElementById('activity-range').value);

  return features.filter(feature => {
    const props = feature.properties;
    return (
      selectedDays.includes(props.day_code) &&
      props.hour_code === hour &&
      props.road_type === selectedRoadType &&
      props.activity_index >= activityThreshold
    );
  });
}

document.getElementById('hour-range').addEventListener('input', fetchAndRenderData);

function loadDailyPoints() {
  fetch('/api/daily_activity')
    .then(res => res.json())
    .then(data => {
      if (map.getLayer('activity-circles')) {
        map.removeLayer('activity-circles');
        map.removeSource('activity-circles');
      }

      map.addSource('activity-circles', { type: 'geojson', data });
      map.addLayer({
        id: 'activity-circles',
        type: 'circle',
        source: 'activity-circles',
        paint: {
          'circle-radius': [
            'interpolate',
            ['linear'],
            ['get', 'activity_index'],
            0, 2,
            0.1, 10,
            1, 30
          ],
          'circle-color': '#FF5733',
          'circle-opacity': 0.6
        }
      });
    });
}
