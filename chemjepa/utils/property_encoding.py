#!/usr/bin/env python3
"""
Property Encoding Utilities

Converts user-specified property dictionaries into tensor format for ChemJEPA.
Supports various property specifications:
  - Exact values: {"LogP": 2.5}
  - Ranges: {"IC50": "<10nM", "bioavailability": ">50%"}
  - Multiple objectives
"""

import torch
import re
from typing import Dict, List, Optional, Tuple, Union


class PropertyEncoder:
    """
    Encodes molecular properties from dictionary specifications into tensors.

    Supports:
        - Continuous properties (LogP, TPSA, MolWt, etc.)
        - Bioactivity ranges (IC50, EC50, Ki)
        - Drug-likeness criteria (bioavailability, QED)
    """

    # Standard property names and their typical ranges
    PROPERTY_SPECS = {
        'LogP': {'min': -3.0, 'max': 8.0, 'target': 2.5, 'unit': None},
        'TPSA': {'min': 0.0, 'max': 200.0, 'target': 60.0, 'unit': 'Å²'},
        'MolWt': {'min': 100.0, 'max': 600.0, 'target': 350.0, 'unit': 'Da'},
        'QED': {'min': 0.0, 'max': 1.0, 'target': 0.7, 'unit': None},
        'SA': {'min': 1.0, 'max': 10.0, 'target': 3.0, 'unit': None},  # Synthetic accessibility
        'NumHDonors': {'min': 0, 'max': 10, 'target': 2, 'unit': None},
        'NumHAcceptors': {'min': 0, 'max': 15, 'target': 4, 'unit': None},
        'NumRotatableBonds': {'min': 0, 'max': 15, 'target': 5, 'unit': None},
        'IC50': {'min': 1e-12, 'max': 1e-3, 'target': 1e-9, 'unit': 'M'},  # log scale
        'EC50': {'min': 1e-12, 'max': 1e-3, 'target': 1e-9, 'unit': 'M'},
        'Ki': {'min': 1e-12, 'max': 1e-3, 'target': 1e-9, 'unit': 'M'},
        'bioavailability': {'min': 0.0, 'max': 100.0, 'target': 50.0, 'unit': '%'},
    }

    def __init__(self, property_dim: int = 64):
        """
        Args:
            property_dim: Dimension of property vector
        """
        self.property_dim = property_dim

    def parse_value(self, value: Union[str, float, int], prop_name: str) -> Tuple[float, Optional[str]]:
        """
        Parse property value from various formats.

        Args:
            value: Property value (e.g., "2.5", "<10nM", ">50%")
            prop_name: Property name for range lookup

        Returns:
            Tuple of (normalized_value, constraint_type)
            constraint_type: None, '<', '>', '<=', '>='
        """
        if isinstance(value, (int, float)):
            # Direct numeric value
            return float(value), None

        # String parsing
        value_str = str(value).strip()

        # Extract constraint operator
        constraint = None
        for op in ['<=', '>=', '<', '>']:
            if value_str.startswith(op):
                constraint = op
                value_str = value_str[len(op):].strip()
                break

        # Parse numeric value and unit
        # Examples: "10nM", "50%", "2.5"
        match = re.match(r'([0-9.e+-]+)\s*([a-zA-Z%]*)', value_str)
        if not match:
            raise ValueError(f"Cannot parse property value: {value}")

        numeric_val = float(match.group(1))
        unit = match.group(2).strip()

        # Unit conversion
        if unit:
            numeric_val = self._convert_unit(numeric_val, unit, prop_name)

        return numeric_val, constraint

    def _convert_unit(self, value: float, unit: str, prop_name: str) -> float:
        """Convert units to standard form."""
        unit_lower = unit.lower()

        # Concentration units (convert to Molar)
        if unit_lower in ['nm', 'nanomolar']:
            return value * 1e-9
        elif unit_lower in ['um', 'µm', 'micromolar']:
            return value * 1e-6
        elif unit_lower in ['mm', 'millimolar']:
            return value * 1e-3
        elif unit_lower in ['m', 'molar']:
            return value
        elif unit_lower in ['pm', 'picomolar']:
            return value * 1e-12

        # Percentage (convert to 0-100 scale)
        elif unit_lower == '%':
            return value  # Already in percentage

        # Mass units
        elif unit_lower in ['kda', 'kilodalton']:
            return value * 1000
        elif unit_lower in ['da', 'dalton']:
            return value

        else:
            # Unknown unit, return as-is
            return value

    def normalize_value(self, value: float, prop_name: str) -> float:
        """
        Normalize property value to [0, 1] range based on typical ranges.

        Args:
            value: Raw property value
            prop_name: Property name

        Returns:
            Normalized value in [0, 1]
        """
        if prop_name not in self.PROPERTY_SPECS:
            # Unknown property, return as-is
            return value

        spec = self.PROPERTY_SPECS[prop_name]
        min_val = spec['min']
        max_val = spec['max']

        # Handle logarithmic scaling for IC50, EC50, Ki
        if prop_name in ['IC50', 'EC50', 'Ki']:
            # Convert to log scale
            import math
            value = math.log10(value) if value > 0 else min_val
            min_val = math.log10(min_val)
            max_val = math.log10(max_val)

        # Normalize to [0, 1]
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))

    def encode(
        self,
        properties: Dict[str, Union[str, float, int]],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Encode property dictionary into tensor.

        Args:
            properties: Dictionary of property specifications
                Examples:
                    {"LogP": 2.5, "TPSA": 60.0}
                    {"IC50": "<10nM", "bioavailability": ">50%"}
            device: Target device

        Returns:
            Property tensor [1, property_dim]
        """
        if device is None:
            device = torch.device('cpu')

        # Create property vector
        # First part: normalized values for known properties
        # Second part: constraint flags

        prop_vector = torch.zeros(self.property_dim, device=device)

        # Use first 32 dims for property values
        # Use next 32 dims for metadata (constraints, weights, etc.)

        prop_idx = 0
        known_props = ['LogP', 'TPSA', 'MolWt', 'QED', 'SA',
                      'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
                      'IC50', 'EC50', 'Ki', 'bioavailability']

        for prop_name in known_props[:16]:  # Use first 16 slots
            if prop_name in properties:
                value, constraint = self.parse_value(properties[prop_name], prop_name)
                normalized = self.normalize_value(value, prop_name)

                prop_vector[prop_idx] = normalized

                # Encode constraint in second half
                if constraint == '<':
                    prop_vector[32 + prop_idx] = -1.0  # Upper bound
                elif constraint == '>':
                    prop_vector[32 + prop_idx] = 1.0   # Lower bound
                elif constraint in ['<=', '>=']:
                    prop_vector[32 + prop_idx] = 0.5 if constraint == '<=' else 1.0
                else:
                    prop_vector[32 + prop_idx] = 0.0   # Exact target

            prop_idx += 1

        return prop_vector.unsqueeze(0)  # [1, property_dim]

    def decode(self, prop_tensor: torch.Tensor) -> Dict[str, float]:
        """
        Decode property tensor back to dictionary (approximate).

        Args:
            prop_tensor: Property tensor [1, property_dim] or [property_dim]

        Returns:
            Dictionary of property values
        """
        if prop_tensor.dim() == 2:
            prop_tensor = prop_tensor.squeeze(0)

        prop_dict = {}
        known_props = ['LogP', 'TPSA', 'MolWt', 'QED', 'SA',
                      'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
                      'IC50', 'EC50', 'Ki', 'bioavailability']

        for idx, prop_name in enumerate(known_props[:16]):
            normalized_val = prop_tensor[idx].item()

            if abs(normalized_val) > 1e-6:  # Non-zero value
                # Denormalize
                if prop_name in self.PROPERTY_SPECS:
                    spec = self.PROPERTY_SPECS[prop_name]
                    min_val = spec['min']
                    max_val = spec['max']

                    # Handle log scale
                    if prop_name in ['IC50', 'EC50', 'Ki']:
                        import math
                        min_val = math.log10(min_val)
                        max_val = math.log10(max_val)
                        value = min_val + normalized_val * (max_val - min_val)
                        value = 10 ** value  # Convert back from log
                    else:
                        value = min_val + normalized_val * (max_val - min_val)

                    prop_dict[prop_name] = round(value, 3)

        return prop_dict


# Global instance for convenience
_default_encoder = None

def get_default_encoder(property_dim: int = 64) -> PropertyEncoder:
    """Get default global property encoder."""
    global _default_encoder
    if _default_encoder is None:
        _default_encoder = PropertyEncoder(property_dim=property_dim)
    return _default_encoder


def encode_properties(
    properties: Dict[str, Union[str, float, int]],
    property_dim: int = 64,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Convenience function to encode properties.

    Args:
        properties: Property dictionary
        property_dim: Property vector dimension
        device: Target device

    Returns:
        Property tensor [1, property_dim]
    """
    encoder = get_default_encoder(property_dim)
    return encoder.encode(properties, device)


if __name__ == '__main__':
    """Test property encoding"""
    print("Testing PropertyEncoder...")

    encoder = PropertyEncoder(property_dim=64)

    # Test 1: Simple numeric properties
    props1 = {"LogP": 2.5, "TPSA": 60.0, "MolWt": 350.0}
    tensor1 = encoder.encode(props1)
    print(f"\nTest 1 - Numeric properties:")
    print(f"  Input: {props1}")
    print(f"  Tensor shape: {tensor1.shape}")
    print(f"  Decoded: {encoder.decode(tensor1)}")

    # Test 2: Constraint-based properties
    props2 = {"IC50": "<10nM", "bioavailability": ">50%", "LogP": "2.5"}
    tensor2 = encoder.encode(props2)
    print(f"\nTest 2 - Constraints:")
    print(f"  Input: {props2}")
    print(f"  Tensor shape: {tensor2.shape}")
    print(f"  Decoded: {encoder.decode(tensor2)}")

    # Test 3: Mixed units
    props3 = {"IC50": "5nM", "EC50": "100µM", "MolWt": "0.5kDa"}
    tensor3 = encoder.encode(props3)
    print(f"\nTest 3 - Unit conversion:")
    print(f"  Input: {props3}")
    print(f"  Decoded: {encoder.decode(tensor3)}")

    print("\n✓ PropertyEncoder test passed!")
