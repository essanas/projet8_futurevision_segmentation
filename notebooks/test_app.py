{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "660731e1-a552-43a6-97b3-d512839d7454",
   "metadata": {},
   "outputs": [],
   "source": [
    "from fastapi import FastAPI\n",
    "from fastapi.responses import FileResponse\n",
    "from pathlib import Path\n",
    "\n",
    "app = FastAPI()\n",
    "\n",
    "@app.get(\"/\")\n",
    "async def serve_html():\n",
    "    path = Path(__file__).parent / \"interface.html\"\n",
    "    return FileResponse(path, media_type=\"text/html\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
