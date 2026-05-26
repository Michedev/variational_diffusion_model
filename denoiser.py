class ImageTransformerDenoiser(nn.Module):
    def __init__(self, in_channels=3, image_size=32, patch_size=4, hidden_dim=256, t_emb_dim=64, num_layers=4):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim))

        self.time_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Linear(hidden_dim, in_channels * patch_size * patch_size)

    def forward(self, x, t_emb):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        p = self.patch_size
        nh, nw = H // p, W // p

        # Patch embedding
        x_emb = self.patch_embed(x)  # [B, hidden_dim, nh, nw]
        x_emb = x_emb.flatten(2).transpose(1, 2)  # [B, N, hidden_dim]

        # Add positions and time
        t_emb_proj = self.time_mlp(t_emb).unsqueeze(1)  # [B, 1, hidden_dim]
        x_emb = x_emb + self.pos_embed + t_emb_proj

        # Transformer
        out = self.transformer(x_emb)  # [B, N, hidden_dim]

        # Unpatchify
        out = self.head(out)  # [B, N, C * p * p]
        out = out.transpose(1, 2).view(B, C, p, p, nh, nw)
        out = out.permute(0, 1, 4, 2, 5, 3).contiguous().view(B, C, H, W)
        return out