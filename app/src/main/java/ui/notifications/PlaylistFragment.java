package ui.notifications;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.Observer;
import androidx.lifecycle.ViewModelProvider;

import com.example.cfm.R;

import android.widget.Button;


import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.example.spotify_framework.Playlist;
import com.example.spotify_framework.Song;


import java.util.ArrayList;
import java.util.List;

public class PlaylistFragment extends Fragment {

    private PlaylistViewModel plistViewModel = null;
    private PlaylistAdapter adapter;

    public PlaylistViewModel getVM(){return plistViewModel;}
    public void setVM(PlaylistViewModel blockSelectViewModel){this.plistViewModel = blockSelectViewModel;}

    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {
        if (plistViewModel == null)
            plistViewModel = new ViewModelProvider(this).get(PlaylistViewModel.class);

        View root = inflater.inflate(R.layout.fragment_notifications, container, false);

        Button add_playlist = root.findViewById(R.id.new_playlist);
        System.out.println(add_playlist);
        add_playlist.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // go to new page
            }
        });
        //observes if there are any modifications to the select list and automatically performs onChanged
        RecyclerView rView = root.findViewById(R.id.listening_history);
        adapter = new PlaylistAdapter(this.getContext(), new ArrayList());
        rView.setAdapter(adapter);
        rView.setLayoutManager(new LinearLayoutManager(this.getContext()));
        plistViewModel.getAllPlaylists().observe(getViewLifecycleOwner(), new Observer<List<Playlist>>() { //observes any changes made to the live data playlist list
            @Override
            public void onChanged(@Nullable List<Playlist> s) {
                if (s == null || s.size() == 0){ // check if the list is empty. if it is, then we want to create a default entry
                    System.out.println("creating a new playlist");
                    Playlist plist = new Playlist("No title", "No Description", new ArrayList<Song>(), new ArrayList<Song>(), null);
                    ArrayList<Playlist> listP = new ArrayList<Playlist>();
                    listP.add(plist);
                    plistViewModel.updateList(listP);

                }
                adapter.setPlaylists(s); // sets the list for the adapter

                //I might also want to change the actual values within the recycler view, but to do that I need to inflate the single_select_recycler
            }
        });
        return root;
    }
    public List<Playlist> getAdapterList(){return adapter.getPlaylists();}
    public PlaylistAdapter getAdapter(){return adapter;}

    // returning from another activity. in this case, creating a new playlist
    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        Log.i("Task frag", "result called!!!!");
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 100 && resultCode > 0) {
            Playlist task = (Playlist) data.getSerializableExtra("task"); // song creation page should return this
            if (resultCode == 1) {
                plistViewModel.insert(task);
            } else if (resultCode == 2) {
                plistViewModel.update(task);
            }
        }
    }


}