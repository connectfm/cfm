package ui.notifications;

import android.app.Application;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.ViewModel;
import java.util.List;
import spotify_framework.Playlist;
import ui.databases.PlaylistRepo;

public class PlaylistViewModel extends ViewModel {

	private final PlaylistRepo repo;
	private final LiveData<List<Playlist>> playlists;


	public PlaylistViewModel() {
		repo = new PlaylistRepo();
		playlists = repo.getAllPlaylists();
	}

	public PlaylistViewModel(Application app) {
		//super(app);
		repo = new PlaylistRepo(app);
		playlists = repo.getAllPlaylists();
	}

	public void updateList(List<Playlist> playlists) {
		repo.setAllPlaylists(playlists);
	}


	public LiveData<List<Playlist>> getAllPlaylists() {
		return playlists;
	}

	public void insert(Playlist plist) {
		repo.insert(plist);
	}

	public void delete(Playlist plist) {
		repo.delete(plist);
	}

	public void update(Playlist plist) {
		repo.update(plist);
	}
}