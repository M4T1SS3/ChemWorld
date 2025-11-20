import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/Table'
import type { MolecularProperties } from '@/lib/types'

interface PropertiesTableProps {
  properties: MolecularProperties
}

export default function PropertiesTable({ properties }: PropertiesTableProps) {
  const propertyRows = [
    { name: 'LogP', value: properties.LogP, description: 'Lipophilicity' },
    { name: 'TPSA', value: properties.TPSA, description: 'Polar Surface Area (Ų)' },
    { name: 'Molecular Weight', value: properties.MolWt, description: 'Da' },
    { name: 'QED', value: properties.QED, description: 'Drug-likeness (0-1)' },
    { name: 'SA', value: properties.SA, description: 'Synthetic Accessibility' },
    { name: 'H-Bond Donors', value: properties.NumHDonors, description: 'Count' },
    { name: 'H-Bond Acceptors', value: properties.NumHAcceptors, description: 'Count' },
    { name: 'Rotatable Bonds', value: properties.NumRotatableBonds, description: 'Count' },
  ]

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Property</TableHead>
          <TableHead>Value</TableHead>
          <TableHead>Description</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {propertyRows.map((row) => (
          <TableRow key={row.name}>
            <TableCell className="font-medium">{row.name}</TableCell>
            <TableCell>
              {row.value !== undefined
                ? typeof row.value === 'number'
                  ? row.value.toFixed(2)
                  : row.value
                : 'N/A'}
            </TableCell>
            <TableCell className="text-text-muted">{row.description}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}
