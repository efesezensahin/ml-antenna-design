"""
CST Studio Suite Otomasyon Sürücüsü (Live CST COM + SQLite Çözücü).
Windows 11 / CST 2019+ uyumludur. Pop-up ve History hatası üretmez.
"""
import os
import sqlite3
import numpy as np
import win32com.client


class CSTDriver:
    def __init__(self, cst_project_path: str):
        self.cst_path = os.path.abspath(cst_project_path)
        self.project_dir = os.path.splitext(self.cst_path)[0]
        self.db_path = os.path.join(self.project_dir, "Result", "Storage.sdb")

        # CST COM Bağlantısı
        print("CST Studio Suite baglantisi kuruluyor...")
        self.cst_app = win32com.client.Dispatch("CSTStudio.Application")
        self.project = self.cst_app.Active3D()

        if self.project is None:
            raise RuntimeError(
                f"CST'de acik 3D proje bulunamadi! Lutfen su projeyi CST ile acin:\n{self.cst_path}"
            )
        print("CST 3D Projesine basariyla baglanildi!")

    def run_simulation(self, param_dict: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Parametreleri dogrudan CST'ye aktarir (AddToHistory olmadan),
        Geometriyi Rebuild eder, Transient Solver'i calistirir.
        """
        # 1. Assign parameters directly using StoreParameter (no pop-up will appear)
        self.project._FlagAsMethod("StoreParameter")
        for k, v in param_dict.items():
            self.project.StoreParameter(k, f"{float(v):.4f}")

        # 2. Update the geometry.
        self.project._FlagAsMethod("Rebuild")
        self.project.Rebuild()

        # 3. Clear old results.
        self.project._FlagAsMethod("DeleteResults")
        self.project.DeleteResults()

        # 4. Start Transient Solver
        solver = self.project.Solver
        solver._FlagAsMethod("Start")
        solver.Start()

        # 5. Save the results to disk (Storage.sdb)
        self.project.Save()

        # 6. Read the latest S11 data.
        return self._read_s11_from_storage()

    def _read_s11_from_storage(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Storage.sdb icindeki en son cS1(1)1(1).sig verisini cozer.
        """
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Storage.sdb bulunamadi: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Merging of Signals and SigData3
        try:
            query = """
            SELECT d.bData 
            FROM Signals s
            JOIN SigData3 d ON s.sig_id = d.sig_id
            WHERE s.name LIKE '%cS1(1)1(1)%' 
            ORDER BY s.sig_id DESC 
            LIMIT 1;
            """
            cursor.execute(query)
            row = cursor.fetchone()
        except sqlite3.OperationalError:
            query = """
            SELECT bData 
            FROM SigData3 
            ORDER BY sig_id DESC 
            LIMIT 1;
            """
            cursor.execute(query)
            row = cursor.fetchone()

        conn.close()

        if not row or not row[0]:
            raise RuntimeError("Storage.sdb icinde S11 sinyali bulunamadi!")

        bdata = row[0]
        n_pts = int(np.frombuffer(bdata[:24], dtype=np.int64)[0])
        freq_bytes = n_pts * 4

        freqs = np.frombuffer(bdata[24 : 24 + freq_bytes], dtype=np.float32)
        s_complex = np.frombuffer(bdata[24 + freq_bytes :], dtype=np.complex64)
        s11_db = 20 * np.log10(np.maximum(np.abs(s_complex), 1e-12))

        return freqs, s11_db