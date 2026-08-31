import torch
import torch.nn as nn

class BBoxEncoder(nn.Module):
    def __init__(
        self,
        num_classes=10,
        class_dim=64,
        geom_dim=64,
        hidden_dim=256,
        out_dim=256,
    ):
        super().__init__()

        self.class_embedding = nn.Embedding(
            num_classes,
            class_dim
        )

        self.geometry_mlp = nn.Sequential(
            nn.Linear(4, geom_dim),
            nn.SiLU(),
            nn.Linear(geom_dim, geom_dim),
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(
                class_dim + geom_dim,
                hidden_dim
            ),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, class_ids, boxes):
        """
        class_ids: [B, N]
        boxes:     [B, N, 4]
                   normalized [cx, cy, w, h]

        returns:
            [B, N, out_dim]
        """

        class_feat = self.class_embedding(class_ids)
        geom_feat = self.geometry_mlp(boxes)

        feat = torch.cat(
            [class_feat, geom_feat],
            dim=-1
        )

        return self.output_mlp(feat)