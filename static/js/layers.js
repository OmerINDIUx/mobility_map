const trafficLayer = {
  id: 'traffic',
  type: 'line',
  source: 'traffic',
  paint: {
    'line-color': [
      'interpolate',
      ['linear'],
      ['get', 'activity_index'],
      0, '#2DC4B2',
      0.02, '#3BB3C3',
      0.04, '#669EC4',
      0.06, '#8B88B6',
      0.08, '#A2719B',
      0.1, '#AA5E79'
    ],
    'line-width': 4
  }
};
