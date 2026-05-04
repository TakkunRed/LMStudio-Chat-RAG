"""
RAG (Retrieval-Augmented Generation) モジュール
- ChromaDB: ベクトルストア（永続化）
- sentence-transformers: 多言語埋め込みモデル（日本語対応）
- 対応形式: .txt, .pdf（拡張可能）
"""

import os
import re
from pathlib import Path
from typing import Optional

# huggingface_hub の警告レベルを error に上げて不要な警告を抑制
# （sentence-transformers のモデルロード前に設定する必要があるためここで行う）
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

import chromadb

# サポート拡張子 → パーサー関数名のマッピング（拡張時はここに追加）
SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


class RAGManager:
    def __init__(
        self,
        db_path: str = "./rag_db",
        collection_name: str = "documents",
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        chunk_size: int = 400,
        chunk_overlap: int = 40,
    ):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # ChromaDB（ディスク永続化）
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # 埋め込みモデル（初回起動時にダウンロード ~450MB）
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    # ── 埋め込み ──────────────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    # ── テキスト分割 ──────────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> list[str]:
        """単語単位でチャンク分割（オーバーラップ付き）"""
        text = re.sub(r"\s+", " ", text).strip()
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end >= len(words):
                break
            start += self.chunk_size - self.chunk_overlap
        return [c for c in chunks if c.strip()]

    # ── ファイル読み込み ──────────────────────────────────────────────────

    def _read_txt(self, file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8", errors="ignore")

    def _read_pdf(self, file_path: Path) -> str:
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            return "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except Exception as e:
            raise ValueError(f"PDF読み込みエラー: {e}") from e

    def _read_file(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()
        if ext == ".txt":
            return self._read_txt(file_path)
        if ext == ".pdf":
            return self._read_pdf(file_path)
        raise ValueError(f"未対応の形式: {ext}")

    # ── インデックス操作 ──────────────────────────────────────────────────

    def _delete_by_source(self, source: str) -> int:
        """同一ファイルの既存チャンクを削除して重複を防ぐ"""
        try:
            results = self.collection.get(where={"source": source})
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                return len(results["ids"])
        except Exception:
            pass
        return 0

    def add_document(self, file_path: str | Path) -> dict:
        """1ファイルをインデックスに追加"""
        file_path = Path(file_path)
        if not file_path.exists():
            return {"success": False, "file": file_path.name, "message": "ファイルが見つかりません"}
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return {"success": False, "file": file_path.name, "message": f"未対応の形式: {file_path.suffix}"}

        try:
            text = self._read_file(file_path)
        except ValueError as e:
            return {"success": False, "file": file_path.name, "message": str(e)}

        if not text.strip():
            return {"success": False, "file": file_path.name, "message": "テキストを抽出できませんでした"}

        chunks = self._chunk_text(text)
        source_key = str(file_path)
        self._delete_by_source(source_key)

        ids = [f"{file_path.name}::{i}" for i in range(len(chunks))]
        embeddings = self._embed(chunks)
        metadatas = [{"source": source_key, "chunk_index": i} for i in range(len(chunks))]

        self.collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        return {"success": True, "file": file_path.name, "chunks": len(chunks)}

    def add_directory(self, dir_path: str | Path) -> list[dict]:
        """ディレクトリ内の全対応ファイルをインデックスに追加"""
        dir_path = Path(dir_path)
        results = []
        for ext in SUPPORTED_EXTENSIONS:
            for fp in dir_path.rglob(f"*{ext}"):
                results.append(self.add_document(fp))
        return results

    def delete_document(self, file_name: str) -> dict:
        """ファイル名でインデックスから削除"""
        try:
            all_items = self.collection.get(include=["metadatas"])
            ids_to_delete = [
                all_items["ids"][i]
                for i, m in enumerate(all_items["metadatas"])
                if Path(m.get("source", "")).name == file_name
            ]
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                return {"success": True, "deleted_chunks": len(ids_to_delete)}
            return {"success": False, "message": "該当ファイルが見つかりません"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def clear(self) -> dict:
        """全チャンクを削除"""
        count = self.collection.count()
        if count > 0:
            all_ids = self.collection.get()["ids"]
            self.collection.delete(ids=all_ids)
        return {"cleared_chunks": count}

    # ── 検索 ──────────────────────────────────────────────────────────────

    def search(self, query: str, n_results: int = 5, min_score: float = 0.3) -> list[dict]:
        """クエリに関連するチャンクを検索"""
        total = self.collection.count()
        if total == 0:
            return []

        results = self.collection.query(
            query_embeddings=self._embed([query]),
            n_results=min(n_results, total),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - dist  # cosine distance → similarity
            if score >= min_score:
                output.append({
                    "content": doc,
                    "source": Path(meta.get("source", "")).name,
                    "score": round(score, 4),
                })
        return output

    # ── 状態確認 ──────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        count = self.collection.count()
        sources: set[str] = set()
        if count > 0:
            for m in self.collection.get(include=["metadatas"])["metadatas"]:
                sources.add(Path(m.get("source", "")).name)
        return {
            "total_chunks": count,
            "total_documents": len(sources),
            "documents": sorted(sources),
        }
